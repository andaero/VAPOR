	>�^���$@>�^���$@!>�^���$@	9T^uk��?9T^uk��?!9T^uk��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6>�^���$@�;��)��?1䞮�X� @A�����?Iж�u�7@Y �8�@d�?*	43333�d@2U
Iterator::Model::ParallelMapV2u���?!<X�N B@)u���?1<X�N B@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceB`��"۩?!�)ZCg>@)B`��"۩?1�)ZCg>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���<,�?!��>�|�'@)X�5�;N�?1�e�UdY$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��K7�A�?!�uTpC@)9��v���?1�9��rN@:Preprocessing2F
Iterator::Model�-����?!�$��!E@)��_�L�?1>.d.�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip鷯猸?!�x�0?�L@)�q����?1�"�b��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap������?!+\r�
�D@)n��t?1�f�a��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǺ���f?!�k����?)Ǻ���f?1�k����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 12.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�65.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no99T^uk��?I�a��S@Q!=�� 4@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�;��)��?�;��)��?!�;��)��?      ��!       "	䞮�X� @䞮�X� @!䞮�X� @*      ��!       2	�����?�����?!�����?:	ж�u�7@ж�u�7@!ж�u�7@B      ��!       J	 �8�@d�? �8�@d�?! �8�@d�?R      ��!       Z	 �8�@d�? �8�@d�?! �8�@d�?b      ��!       JGPUY9T^uk��?b q�a��S@y!=�� 4@