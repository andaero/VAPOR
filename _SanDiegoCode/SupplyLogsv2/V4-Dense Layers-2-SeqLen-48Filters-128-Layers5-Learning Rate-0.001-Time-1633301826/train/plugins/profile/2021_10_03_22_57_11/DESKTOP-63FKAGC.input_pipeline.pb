	�,AF@� @�,AF@� @!�,AF@� @	/�F�ym�?/�F�ym�?!/�F�ym�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�,AF@� @�h9�C��?18�Jw�Y @A�U�p�?I�tx�g@Y��`�.�?*effff�W@)      =2U
Iterator::Model::ParallelMapV2�:pΈҞ?!�E]t�?@)�:pΈҞ?1�E]t�?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��ZӼ�?!�M�䁐5@)�5�;Nё?1k3	�d2@:Preprocessing2F
Iterator::Model��&��?!����)kH@)?�ܵ�|�?1d6��1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate䃞ͪϕ?!~o�'�6@)���QI�?1��#�;.@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey�&1�|?!�L��*�@)y�&1�|?1�L��*�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��@��Ǩ?!fMYS֔I@)HP�s�r?1����Gs@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�~j�t�h?!)�8�^	@)�~j�t�h?1)�8�^	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapg��j+��?!uh��X�8@)�J�4a?1�ǧ�L�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 12.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�61.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9.�F�ym�?ILu㚚�R@Q��]���8@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�h9�C��?�h9�C��?!�h9�C��?      ��!       "	8�Jw�Y @8�Jw�Y @!8�Jw�Y @*      ��!       2	�U�p�?�U�p�?!�U�p�?:	�tx�g@�tx�g@!�tx�g@B      ��!       J	��`�.�?��`�.�?!��`�.�?R      ��!       Z	��`�.�?��`�.�?!��`�.�?b      ��!       JGPUY.�F�ym�?b qLu㚚�R@y��]���8@