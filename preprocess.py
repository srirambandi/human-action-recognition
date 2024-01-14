def sample_fn_sorted_loop(n_frames, inv_sampl_rate, img_size, is_training,
                          do_random_crop, flow=False):
  """Pad video."""
  # n_frames -- number of consecutive frames to sample per video
  def my_fun(frames):
    """."""
    # sample n_frames among frames

    actual_n_frames = tf.shape(frames)[0]
    sel_idx = tf.range(actual_n_frames)

    # repeat video enough times to get n_frames*inv_sampl_rate
    n_repeats = tf.to_int32(tf.ceil(tf.div(tf.to_float(inv_sampl_rate*n_frames),
                                           tf.to_float(actual_n_frames))))

    sel_idx = tf.tile(sel_idx, [n_repeats])

    the_range = tf.range(inv_sampl_rate*n_frames, delta=inv_sampl_rate)

    if do_random_crop:
      the_range_train = _random_crop_range(sel_idx, [inv_sampl_rate * n_frames],
                                           (tf.shape(sel_idx)[0] -
                                            (inv_sampl_rate*n_frames) + 1))
      the_range = tf.gather(the_range_train, the_range)

    sel_idx_small = tf.gather(sel_idx, the_range)

    frames = tf.gather(frames, sel_idx_small)

    if is_training:
      fn = _video_preprocess(n_frames, img_size, img_size, is_training, is_flow=flow)
    else:
      fn = _video_preprocess(n_frames, img_size, img_size, is_training, is_flow=flow)

    frames = fn(frames)

    return frames

  return my_fun





def _random_crop_range(value, size, max_first_id, seed=None, name=None):
  """Randomly crops a tensor to a given size; first element in [0,max_first_id].

  Slices a shape `size` portion out of `value` at a uniformly chosen offset.
  Requires `value.shape >= size`.

  If a dimension should not be cropped, pass the full size of that dimension.
  For example, RGB images can be cropped with
  `size = [crop_height, crop_width, 3]`.

  Args:
    value: Input tensor to crop.
    size: 1-D tensor with size the rank of `value`.
    max_first_id: Python integer.
    seed: Python integer. Used to create a random seed. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    name: A name for this operation (optional).

  Returns:
    A cropped tensor of the same rank as `value` and shape `size`.
  """
  with tf.name_scope(name, 'random_crop_range', [value, size]) as name:
    value = tf.convert_to_tensor(value, name='value')
    size = tf.convert_to_tensor(size, dtype=tf.int32, name='size')
    max_first_id = tf.convert_to_tensor(max_first_id, dtype=tf.int32,
                                        name='max_first_id')
    shape = tf.shape(value)
    check = tf.Assert(
        tf.reduce_all(shape >= size),
        ['Need value.shape >= size, got ', shape, size])
    shape = tf.with_dependencies([check], shape)
    limit = max_first_id
    offset = tf.random_uniform(
        tf.shape(shape),
        dtype=size.dtype,
        maxval=size.dtype.max,
        seed=seed) % limit
    return tf.slice(value, offset, size, name=name)







def _video_preprocess(num_frames, height, width, is_training, flip=[],
                      is_flow=False):
  """."""
  def the_fun(video):
    """."""
    # preprocess the frames in a video consistently
    size = tf.minimum(height, width)

    ### resize to minimum size using tim's code
    height_ = tf.to_float(tf.shape(video)[1])
    width_ = tf.to_float(tf.shape(video)[2])
    size_ = tf.reshape(tf.to_float(size), [1])

    min_height, min_width = tf.cond(
        height_ > width_,
        lambda: (tf.reshape(height_ * (size_ / width_), [1]), size_),
        lambda: (size_, tf.reshape(width_ * (size_ / height_), [1])),
    )

    new_shape = tf.to_int32(tf.concat([min_height*1.15, min_width*1.15], 0))

    images = tf.unstack(video)

    for i in range(0, len(images)):
      images[i] = tf.image.convert_image_dtype(images[i], dtype=tf.float32)
      images[i] = tf.expand_dims(images[i], 0)
      images[i] = tf.image.resize_bilinear(images[i], size=new_shape)
      images[i] = tf.squeeze(images[i], [0])

      # get between -1 and 1
      if not is_flow:
        images[i] = tf.subtract(images[i], 0.5)
        images[i] = tf.multiply(images[i], 2.0)

    if is_training:
      # could apply some distortion here

      images = slim.preprocess.flip_dim(images)

      if is_flow:
        # multiply horizontal component by -1 if flipped
        condition = images[-1]  # is it flipped ?

        # flow has 3 channels, 1 is horizontal, 2 is vertical, 3 is nothing ?
        m = tf.expand_dims(tf.expand_dims([-1.0, 1.0, 1.0], 0), 0)
        images[0] = tf.cond(condition, lambda: tf.multiply(images[0], m),
                            lambda: images[0])

      # remove flipped bool
      del images[-1]

      images = slim.preprocess.random_crop(images, height, width)
    else:
      images = slim.preprocess.central_crop(images, height, width)

    n_channs = 3
    #if is_flow:
    #  n_channs = 2

    images = tf.stack(images)
    images.set_shape([num_frames, height, width, n_channs])

    return images

  return the_fun
