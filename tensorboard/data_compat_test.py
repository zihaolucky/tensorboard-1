# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import test
from tensorflow.python.client import session
from tensorflow.core.framework import summary_pb2
from tensorflow.python.summary import summary as summary_lib
from tensorflow.python.random_ops import random_normal, random_uniform
from tensorflow.python.framework import constant_op
from tensorflow.python.framework.tensor_util import MakeNdarray as make_ndarray

from tensorboard import data_compat
from tensorboard.plugins.histogram import summary as histogram_summary
from tensorboard.plugins.histogram import metadata as histogram_metadata


class MigrateValueTest(test.TestCase):
  """Tests for `migrate_value`.

  These tests should ensure that all first-party new-style values are
  passed through unchanged, that all supported old-style values are
  converted to new-style values, and that other old-style values are
  passed through unchanged.
  """

  def _value_from_op(self, op):
    with session.Session() as sess:
      summary_pbtxt = sess.run(op)
    summary = summary_pb2.Summary()
    summary.ParseFromString(summary_pbtxt)
    # There may be multiple values (e.g., for an image summary that emits
    # multiple images in one batch). That's fine; we'll choose any
    # representative value, assuming that they're homogeneous.
    assert summary.value
    return summary.value[0]

  def _assert_noop(self, value):
    original_pbtxt = value.SerializeToString()
    result = data_compat.migrate_value(value)
    self.assertEqual(value, result)
    self.assertEqual(original_pbtxt, value.SerializeToString())

  def test_scalar(self):
    op = summary_lib.scalar('important_constants', constant_op.constant(0x5f3759df))
    value = self._value_from_op(op)
    assert value.HasField('simple_value'), value
    self._assert_noop(value)

  def test_image(self):
    op = summary_lib.image('mona_lisa',
                           random_normal(shape=[1, 400, 200, 3]))
    value = self._value_from_op(op)
    assert value.HasField('image'), value
    self._assert_noop(value)

  def test_audio(self):
    op = summary_lib.audio('white_noise',
                          random_uniform(shape=[1, 44100],
                                            minval=-1.0,
                                            maxval=1.0),
                          sample_rate=44100)
    value = self._value_from_op(op)
    assert value.HasField('audio'), value
    self._assert_noop(value)

  def test_text(self):
    op = summary_lib.text('lorem_ipsum', constant_op.constant('dolor sit amet'))
    value = self._value_from_op(op)
    assert value.HasField('tensor'), value
    self._assert_noop(value)

  def test_fully_populated_tensor(self):
    metadata = summary_pb2.SummaryMetadata()
    metadata.plugin_data.add(plugin_name='font_of_wisdom',
                             content='adobe_garamond')
    op = summary_lib.tensor_summary(
        name='tensorpocalypse',
        tensor=constant_op.constant([[0.0, 2.0], [float('inf'), float('nan')]]),
        display_name='TENSORPOCALYPSE',
        summary_description='look on my works ye mighty and despair',
        summary_metadata=metadata)
    value = self._value_from_op(op)
    assert value.HasField('tensor'), value
    self._assert_noop(value)

  def test_histogram(self):
    old_op = summary_lib.histogram('important_data',
                                   random_normal(shape=[23, 45]))
    old_value = self._value_from_op(old_op)
    assert old_value.HasField('histo'), old_value
    new_value = data_compat.migrate_value(old_value)

    self.assertEqual('important_data', new_value.tag)
    expected_metadata = histogram_metadata.create_summary_metadata(
        display_name='important_data', description='')
    self.assertEqual(expected_metadata, new_value.metadata)
    self.assertTrue(new_value.HasField('tensor'))
    buckets = make_ndarray(new_value.tensor)
    self.assertEqual(old_value.histo.min, buckets[0][0])
    self.assertEqual(old_value.histo.max, buckets[-1][1])
    self.assertEqual(23 * 45, buckets[:, 2].astype(int).sum())

  def test_new_style_histogram(self):
    op = histogram_summary.op('important_data',
                              random_normal(shape=[10, 10]),
                              bucket_count=100,
                              display_name='Important data',
                              description='secrets of the universe')
    value = self._value_from_op(op)
    assert value.HasField('tensor'), value
    self._assert_noop(value)


if __name__ == '__main__':
  test.main()
