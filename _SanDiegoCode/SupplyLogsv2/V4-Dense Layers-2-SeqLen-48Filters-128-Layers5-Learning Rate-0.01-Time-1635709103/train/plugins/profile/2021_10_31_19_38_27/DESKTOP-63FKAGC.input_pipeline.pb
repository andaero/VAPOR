	��7!@��7!@!��7!@	��^z�[ @��^z�[ @!��^z�[ @"z
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails9��7!@�P�yr?1�Ry;��@A{��>��?I��&3�V@YA�9w�^�?r0*	�G�zdV@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatpw�n�М?! �_�j?@)��y�]��?1��?��	=@:Preprocessing2T
Iterator::Root::ParallelMapV2�V`��?!�vhm�8@)�V`��?1�vhm�8@:Preprocessing2E
Iterator::Root+5{��?!�T*C�C@)��?�Ŋ�?1+c�1�,@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate��ԕϒ?!�^PT�4@)��A�f�?1s��mU'@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice*��g\8�?!����:�!@)*��g\8�?1����:�!@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip����ī?!)�ռEN@)	3m��Js?1 U���@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapMg'���?!�!���7@)cb�qm�h?1i�k�M�
@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorP6�
�ra?!]�J�3@)P6�
�ra?1]�J�3@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 2.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�71.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��^z�[ @I��w�:�Q@Q��Ժ�:@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�P�yr?�P�yr?!�P�yr?      ��!       "	�Ry;��@�Ry;��@!�Ry;��@*      ��!       2	{��>��?{��>��?!{��>��?:	��&3�V@��&3�V@!��&3�V@B      ��!       J	A�9w�^�?A�9w�^�?!A�9w�^�?R      ��!       Z	A�9w�^�?A�9w�^�?!A�9w�^�?b      ��!       JGPUY��^z�[ @b q��w�:�Q@y��Ժ�:@