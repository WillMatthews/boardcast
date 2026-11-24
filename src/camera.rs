//! The real webcam (MJPEG capture + controls) and v4l2loopback output sinks.

use std::fs::File;
use std::io::{Cursor, Write};
use std::os::fd::{BorrowedFd, OwnedFd};

use anyhow::{Context, Result, bail};
use v4l::buffer::Type;
use v4l::control::{Control, Value};
use v4l::format::FourCC;
use v4l::io::mmap::Stream;
use v4l::io::traits::CaptureStream;
use v4l::video::{Capture, Output};
use v4l::{Device, Format};
use zune_jpeg::JpegDecoder;

use crate::image::Rgb;

const CID_FOCUS_ABSOLUTE: u32 = 0x009a_090a;
const CID_FOCUS_AUTO_CONTINUOUS: u32 = 0x009a_090c;

pub enum Focus {
    Auto,
    Fixed(i64),
}

pub struct Camera {
    stream: Stream<'static>,
    /// Time spent decoding the last frame (the rest of `next_into` is waiting for the camera).
    pub last_decode: std::time::Duration,
    pub w: usize,
    pub h: usize,
}

impl Camera {
    pub fn open(path: &str, w: usize, h: usize, focus: Focus) -> Result<Self> {
        let dev = Device::with_path(path).with_context(|| format!("opening {path}"))?;
        let fmt = Capture::set_format(&dev,
            &Format::new(w as u32, h as u32, FourCC::new(b"MJPG")))
            .with_context(|| {
                format!("setting MJPEG {w}x{h} on {path} (is another app using the camera? try `fuser {path}`)")
            })?;
        if fmt.fourcc != FourCC::new(b"MJPG") || (fmt.width, fmt.height) != (w as u32, h as u32) {
            bail!(
                "{path} gave {} {}x{}, wanted MJPG {w}x{h}",
                fmt.fourcc,
                fmt.width,
                fmt.height
            );
        }
        match focus {
            Focus::Auto => set(&dev, CID_FOCUS_AUTO_CONTINUOUS, 1),
            Focus::Fixed(v) => {
                // Autofocus locks onto whoever is in the foreground, not the board, so pin it.
                set(&dev, CID_FOCUS_AUTO_CONTINUOUS, 0);
                set(&dev, CID_FOCUS_ABSOLUTE, v);
            }
        }
        // The stream keeps the device handle alive; leak the Device so the stream can be 'static.
        let dev: &'static Device = Box::leak(Box::new(dev));
        let stream =
            Stream::with_buffers(dev, Type::VideoCapture, 4).context("starting capture stream")?;
        Ok(Camera {
            stream,
            last_decode: Default::default(),
            w,
            h,
        })
    }

    /// Block for the next frame and decode it into `out`.
    pub fn next_into(&mut self, out: &mut Rgb) -> Result<()> {
        let (buf, meta) = CaptureStream::next(&mut self.stream).context("reading frame")?;
        let t0 = std::time::Instant::now();
        let jpeg = &buf[..meta.bytesused as usize];
        let mut dec = JpegDecoder::new(Cursor::new(jpeg));
        dec.decode_headers().context("bad MJPEG frame")?;
        let (w, h) = dec.dimensions().context("no dimensions")?;
        if (w, h) != (self.w, self.h) {
            bail!("frame is {w}x{h}, expected {}x{}", self.w, self.h);
        }
        out.w = w;
        out.h = h;
        out.data.resize(w * h * 3, 0);
        dec.decode_into(&mut out.data)
            .context("decoding MJPEG frame")?;
        self.last_decode = t0.elapsed();
        Ok(())
    }
}

fn set(dev: &Device, id: u32, v: i64) {
    if let Err(e) = dev.set_control(Control {
        id,
        value: Value::Integer(v),
    }) {
        eprintln!("warning: couldn't set camera control {id:#x}={v}: {e}");
    }
}

/// How frames are handed to v4l2loopback.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum SinkIo {
    /// mmap streaming (QBUF/DQBUF).
    Mmap,
    /// Plain write() calls. On v4l2loopback 0.12.7 with exclusive_caps this leaves the device unusable
    /// after the writer exits (needs a module reload).
    Write,
}

enum SinkOut {
    Mmap(RingWriter),
    Write(File),
}

/// mmap output that fills buffers strictly round-robin.
///
/// v4l2loopback shares buffer memory with readers and returns an output buffer as done the moment it is
/// queued, so "dequeue and refill" (what `v4l::io::mmap::Stream` does) overwrites the very frame a reader is
/// still copying: horizontal tearing. Writing the buffer queued longest ago gives readers N-1 frames of slack.
struct RingWriter {
    fd: i32,
    bufs: Vec<(*mut u8, usize)>,
    next: usize,
    streaming: bool,
    _dev: Device,
}

impl RingWriter {
    fn new(dev: Device, want: u32) -> std::io::Result<Self> {
        use v4l::v4l_sys::*;
        let fd = dev.handle().fd();
        let mut req = v4l2_requestbuffers {
            count: want,
            type_: Type::VideoOutput as u32,
            memory: v4l::memory::Memory::Mmap as u32,
            ..unsafe { std::mem::zeroed() }
        };
        unsafe {
            v4l::v4l2::ioctl(
                fd,
                v4l::v4l2::vidioc::VIDIOC_REQBUFS,
                &mut req as *mut _ as *mut _,
            )?
        };
        let mut w = RingWriter {
            fd,
            bufs: Vec::new(),
            next: 0,
            streaming: false,
            _dev: dev,
        };
        for index in 0..req.count {
            let mut b = Self::desc(index);
            unsafe {
                v4l::v4l2::ioctl(
                    fd,
                    v4l::v4l2::vidioc::VIDIOC_QUERYBUF,
                    &mut b as *mut _ as *mut _,
                )?;
                let ptr = v4l::v4l2::mmap(
                    std::ptr::null_mut(),
                    b.length as usize,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    fd,
                    b.m.offset as libc::off_t,
                )?;
                w.bufs.push((ptr as *mut u8, b.length as usize));
            }
        }
        if w.bufs.len() < 2 {
            return Err(std::io::Error::other(
                "v4l2loopback gave fewer than 2 buffers",
            ));
        }
        Ok(w)
    }

    fn desc(index: u32) -> v4l::v4l_sys::v4l2_buffer {
        v4l::v4l_sys::v4l2_buffer {
            index,
            type_: Type::VideoOutput as u32,
            memory: v4l::memory::Memory::Mmap as u32,
            ..unsafe { std::mem::zeroed() }
        }
    }

    fn write(&mut self, frame: &[u8]) -> std::io::Result<()> {
        use v4l::v4l2::{ioctl, vidioc};
        let i = self.next;
        let (ptr, len) = self.bufs[i];
        if frame.len() > len {
            return Err(std::io::Error::other("frame larger than loopback buffer"));
        }
        // SAFETY: the mapping is `len` bytes, lives until drop, and we own the buffer: v4l2loopback hands
        // output buffers back immediately, and we dequeue after every queue below.
        unsafe { std::ptr::copy_nonoverlapping(frame.as_ptr(), ptr, frame.len()) };
        let mut b = Self::desc(i as u32);
        b.bytesused = frame.len() as u32;
        b.field = 1; // V4L2_FIELD_NONE
        unsafe {
            ioctl(self.fd, vidioc::VIDIOC_QBUF, &mut b as *mut _ as *mut _)?;
            if !self.streaming {
                let mut t = Type::VideoOutput as u32;
                ioctl(self.fd, vidioc::VIDIOC_STREAMON, &mut t as *mut _ as *mut _)?;
                self.streaming = true;
            }
            // Take a buffer back so the driver's queue never fills; which one doesn't matter, we keep
            // writing in ring order regardless.
            let mut d = Self::desc(0);
            if let Err(e) = ioctl(self.fd, vidioc::VIDIOC_DQBUF, &mut d as *mut _ as *mut _) {
                if e.kind() != std::io::ErrorKind::WouldBlock {
                    return Err(e);
                }
            }
        }
        self.next = (i + 1) % self.bufs.len();
        Ok(())
    }
}

impl Drop for RingWriter {
    fn drop(&mut self) {
        unsafe {
            if self.streaming {
                let mut t = Type::VideoOutput as u32;
                let _ = v4l::v4l2::ioctl(
                    self.fd,
                    v4l::v4l2::vidioc::VIDIOC_STREAMOFF,
                    &mut t as *mut _ as *mut _,
                );
            }
            for &(ptr, len) in &self.bufs {
                let _ = v4l::v4l2::munmap(ptr as *mut _, len);
            }
        }
    }
}

/// A v4l2loopback device we write YUYV frames into.
pub struct Sink {
    pub path: String,
    out: SinkOut,
    pub w: usize,
    pub h: usize,
}

impl Sink {
    pub fn open(path: &str, w: usize, h: usize, io: SinkIo) -> Result<Self> {
        let dev = Device::with_path(path)
            .with_context(|| format!("opening {path} - is v4l2loopback loaded? (see README)"))?;
        let caps = dev.query_caps()?;
        if caps.driver != "v4l2 loopback" {
            bail!(
                "{path} is a '{}' device ({}), not v4l2loopback - refusing to write to it",
                caps.driver,
                caps.card
            );
        }
        let want = Format::new(w as u32, h as u32, FourCC::new(b"YUYV"));
        let got = Output::set_format(&dev, &want).with_context(|| {
            format!(
                "setting YUYV {w}x{h} on {path} (device caps {:#x}; if this fails after a previous run, \
                 v4l2loopback is stuck - reload the module, see README)",
                caps.capabilities.bits()
            )
        })?;
        if (got.width, got.height, got.fourcc) != (want.width, want.height, want.fourcc) {
            bail!(
                "{path} accepted {} {}x{} instead of YUYV {w}x{h} (is another writer already using it?)",
                got.fourcc,
                got.width,
                got.height
            );
        }
        let out = match io {
            SinkIo::Mmap => SinkOut::Mmap(
                RingWriter::new(dev, 8)
                    .with_context(|| format!("mapping output buffers on {path}"))?,
            ),
            SinkIo::Write => {
                // Write through a dup of the same fd the format was set on.
                let fd: OwnedFd =
                    unsafe { BorrowedFd::borrow_raw(dev.handle().fd()) }.try_clone_to_owned()?;
                SinkOut::Write(File::from(fd))
            }
        };
        Ok(Sink {
            path: path.to_string(),
            out,
            w,
            h,
        })
    }

    pub fn write(&mut self, yuyv: &[u8]) -> Result<()> {
        debug_assert_eq!(yuyv.len(), self.w * self.h * 2);
        match &mut self.out {
            SinkOut::Write(f) => f.write_all(yuyv),
            SinkOut::Mmap(ring) => ring.write(yuyv),
        }
        .with_context(|| format!("writing to {}", self.path))
    }
}
