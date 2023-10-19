import itertools
import multiprocessing

import PIL.Image
import cv2
import imageio
import more_itertools
import tensorflow as tf


def image_files(
        image_paths, extra_data=None, internal_queue_size=None, batch_size=64, prefetch_gpu=1,
        tee_cpu=False, frame_preproc_fn=None, frame_preproc_size_fn=None,
        varying_resolutions=False):
    if varying_resolutions:
        width, height = None, None
    else:
        width, height = image_extents(image_paths[0])
        if frame_preproc_size_fn is not None:
            width, height = frame_preproc_size_fn(width, height)

    return image_dataset_from_queue(
        images_from_paths_gen, args=(image_paths,), imshape=[height, width], extra_data=extra_data,
        internal_queue_size=internal_queue_size, batch_size=batch_size, prefetch_gpu=prefetch_gpu,
        tee_cpu=tee_cpu, frame_preproc_fn=frame_preproc_fn)


def video_file(
        video_path, extra_data=None, internal_queue_size=None, batch_size=64, prefetch_gpu=1,
        tee_cpu=False, video_slice=slice(None), frame_preproc_fn=None, frame_preproc_size_fn=None):
    width, height = video_extents(video_path)
    if frame_preproc_size_fn is not None:
        width, height = frame_preproc_size_fn(width, height)

    return image_dataset_from_queue(
        sliced_reader, args=(video_path, video_slice), imshape=[height, width],
        extra_data=extra_data, internal_queue_size=internal_queue_size, batch_size=batch_size,
        prefetch_gpu=prefetch_gpu, tee_cpu=tee_cpu, frame_preproc_fn=frame_preproc_fn)


def webcam(
        capture_id=0, extra_data=None, internal_queue_size=None, batch_size=64, prefetch_gpu=1,
        tee_cpu=True):
    cap = cv2.VideoCapture(capture_id)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return image_dataset_from_queue(
        frames_from_webcam, args=(capture_id,), imshape=[height, width],
        extra_data=extra_data, internal_queue_size=internal_queue_size, batch_size=batch_size,
        prefetch_gpu=prefetch_gpu, tee_cpu=tee_cpu)


def youtube(
        url, extra_data=None, internal_queue_size=None, batch_size=64, prefetch_gpu=1,
        tee_cpu=True):
    import pafy
    best_stream = pafy.new(url).getbest()
    width, height = best_stream.dimensions
    return image_dataset_from_queue(
        frames_of_youtube_stream, args=(best_stream.url,), imshape=[height, width],
        extra_data=extra_data, internal_queue_size=internal_queue_size, batch_size=batch_size,
        prefetch_gpu=prefetch_gpu, tee_cpu=tee_cpu)


def interleaved_video_files(
        video_paths, extra_data=None, internal_queue_size=None, batch_size=64, prefetch_gpu=1,
        tee_cpu=False, video_slice=slice(None), frame_preproc_fn=None, frame_preproc_size_fn=None):
    width, height = video_extents(video_paths[0])
    if frame_preproc_size_fn is not None:
        width, height = frame_preproc_size_fn(width, height)
    return image_dataset_from_queue(
        interleaved_frame_gen, args=(video_paths, video_slice), imshape=[height, width],
        extra_data=extra_data, internal_queue_size=internal_queue_size, batch_size=batch_size,
        prefetch_gpu=prefetch_gpu, tee_cpu=tee_cpu, frame_preproc_fn=frame_preproc_fn)


def sliced_reader(path, video_slice):
    video = imageio.get_reader(path, 'ffmpeg', output_params=['-map', '0:v:0'])
    return itertools.islice(video, video_slice.start, video_slice.stop, video_slice.step)


def frames_from_webcam(capture_id):
    cap = cv2.VideoCapture(capture_id)
    while (frame_bgr := cap.read()[1]) is not None:
        yield frame_bgr[..., ::-1]
    cap.release()


def frames_of_youtube_stream(internal_url):
    cap = cv2.VideoCapture(internal_url, cv2.CAP_FFMPEG)
    while (frame_bgr := cap.read()[1]) is not None:
        yield frame_bgr[..., ::-1]


def interleaved_frame_gen(video_paths, video_slice):
    video_readers = [sliced_reader(p, video_slice) for p in video_paths]
    yield from roundrobin(video_readers, [1] * len(video_readers))


def images_from_paths_gen(paths):
    for path in paths:
        yield cv2.imread(path)[..., ::-1]


def image_dataset_from_queue(
        generator_fn, imshape, extra_data, internal_queue_size=None, batch_size=64, prefetch_gpu=1,
        tee_cpu=False, frame_preproc_fn=None, args=None, kwargs=None):
    if internal_queue_size is None:
        internal_queue_size = batch_size * 2 if batch_size is not None else 64

    q = multiprocessing.Queue(internal_queue_size)
    t = multiprocessing.Process(
        target=queue_filler_process, args=(generator_fn, q, args, kwargs))
    t.start()
    if frame_preproc_fn is None:
        frame_preproc_fn = lambda x: x

    def queue_reader():
        while (frame := q.get()) is not None:
            yield frame_preproc_fn(frame)

    frames = queue_reader()
    if tee_cpu:
        frames, frames2 = itertools.tee(frames, 2)
    else:
        frames2 = itertools.repeat(None)

    ds = tf.data.Dataset.from_generator(lambda: frames, tf.uint8, [*imshape[:2], 3])

    if extra_data is not None:
        ds = tf.data.Dataset.zip((ds, extra_data))

    if batch_size is not None:
        ds = ds.batch(batch_size)
    if prefetch_gpu:
        ds = ds.apply(tf.data.experimental.prefetch_to_device('GPU:0', prefetch_gpu))

    if batch_size is not None:
        frames2 = more_itertools.chunked(frames2, batch_size)

    return ds, frames2


def queue_filler_process(generator_fn, q, args, kwargs):
    args = () if args is None else args
    kwargs = {} if kwargs is None else kwargs
    for item in generator_fn(*args, **kwargs):
        q.put(item)
    q.put(None)


def frames_from_video(video_path):
    import ffmpeg
    import subprocess
    import numpy as np
    probe = ffmpeg.probe(video_path)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    target_height = None
    if target_height:
        width = int(width * target_height / height)
        height = target_height

    stream = ffmpeg.input(video_path)
    if target_height:
        stream = stream.filter('scale', size=f'{width}:{height}', flags='area')

    args = stream.output('pipe:', format='rawvideo', pix_fmt='rgb24').compile()
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    while read_bytes := proc.stdout.read(width * height * 3):
        yield np.frombuffer(read_bytes, np.uint8).reshape([height, width, 3])
    proc.wait()


def roundrobin(iterables, sizes):
    iterators = [iter(iterable) for iterable in iterables]
    for iterator, size in zip(itertools.cycle(iterators), itertools.cycle(sizes)):
        for _ in range(size):
            try:
                yield next(iterator)
            except StopIteration:
                return


def video_extents(filepath):
    """Returns the video (width, height) as a numpy array, without loading the pixel data."""

    with imageio.get_reader(filepath, 'ffmpeg') as reader:
        return *reader.get_meta_data()['source_size'],


def image_extents(filepath):
    """Returns the image (width, height) as a numpy array, without loading the pixel data."""

    with PIL.Image.open(filepath) as im:
        return im.size
