	#�g]�� @#�g]�� @!#�g]�� @      ��!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0#�g]�� @ƣT�z�?1����@A�A��v��?I�tp�X@r0*	�x�&1da@2T
Iterator::Root::ParallelMapV2�52;��?!+�Z̫�?@)�52;��?1+�Z̫�?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipD����9�?!���,�J@)���KqU�?1�?�9$�1@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat���+�?!X�˝��1@)���9̗?1�G�$�0@:Preprocessing2E
Iterator::Root ��Ud�?!xEL��G@){�G�z�?1�G{X��,@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenatefI��Z��?!����4�/@)}\*���?1,3Tܢ#@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceP6�
�r�?!�i̱~@)P6�
�r�?1�i̱~@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap�B��f�?!"t�/��2@)��H�}m?1��X�-�@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�N^�U?!�'�P���?)�N^�U?1�'�P���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�71.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��<�U[R@Q�!ө�:@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	ƣT�z�?ƣT�z�?!ƣT�z�?      ��!       "	����@����@!����@*      ��!       2	�A��v��?�A��v��?!�A��v��?:	�tp�X@�tp�X@!�tp�X@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��<�U[R@y�!ө�:@