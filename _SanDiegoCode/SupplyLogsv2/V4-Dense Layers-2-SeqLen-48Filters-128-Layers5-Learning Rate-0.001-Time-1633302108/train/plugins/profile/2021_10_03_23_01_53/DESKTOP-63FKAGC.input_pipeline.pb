	�u���"@�u���"@!�u���"@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�u���"@��V�c��?1z����@A���T�-�?Ir��>sf@*	������S@2U
Iterator::Model::ParallelMapV2U���N@�?!�'Jv�7@)U���N@�?1�'Jv�7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�g��s��?!+�2�C�:@)�:pΈ�?1M�Ь��6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateZd;�O��?!�!���<@)� �	��?1�
kB=e3@:Preprocessing2F
Iterator::ModelB>�٬��?!-"�t7�A@)��~j�t�?1�8H���'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!?!2�ls�$#@)ŏ1w-!?12�ls�$#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��y�)�?!�n�E�P@)HP�s�r?1sNc~,@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora��+ei?!�C�;@)a��+ei?1�C�;@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapc�ZB>�?!�H��-�?@)HP�s�b?1sNc~,@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 13.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�61.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI���[S@Q�y���7@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��V�c��?��V�c��?!��V�c��?      ��!       "	z����@z����@!z����@*      ��!       2	���T�-�?���T�-�?!���T�-�?:	r��>sf@r��>sf@!r��>sf@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q���[S@y�y���7@