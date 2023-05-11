import ffmpeg

# stream = ffmpeg.input('data/video/bread2_3.mp4')
# print('x')

# stream = ffmpeg.hflip(stream)
# print('xx')
# stream = ffmpeg.filter(stream, 'fps', fps=25, round='up')
# print('xxx')
# stream = ffmpeg.trim(start_frame=10, end_frame=20)
# stream = ffmpeg.output(stream, 'data/video/output.mp4')
# print('xxxx')
# ffmpeg.run(stream)
# print('xxxxx')



split = (
    ffmpeg
    .input('data/video/bread2_3.mp4')
    .filter('fps', fps=25, round= 'up')
    .filter_multi_output('split')  # or `.split()`
)
(
    ffmpeg
    .concat(split[0], split[1])
    .output('out.mp4')
    .run()
)
