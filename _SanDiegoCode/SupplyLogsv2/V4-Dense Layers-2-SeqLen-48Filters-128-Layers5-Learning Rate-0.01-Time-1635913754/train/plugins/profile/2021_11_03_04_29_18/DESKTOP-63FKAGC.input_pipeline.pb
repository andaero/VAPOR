	���O�!@���O�!@!���O�!@      ��!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0���O�!@�Ws�`�n?1H3Mg�@A�;��)t�?I������@r0*	v�V�W@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatɎ�@���?!�U�
kD@@)�c�ZB�?1$Jd�?@:Preprocessing2T
Iterator::Root::ParallelMapV2�3��k�?!��M��n3@)�3��k�?1��M��n3@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�R	O���?!���ٚ8@)�8�	�ʌ?1|[U,�-@:Preprocessing2E
Iterator::Root���B�i�?!g�h	;?@)|,}���?11�pƱ�'@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice7l[�� �?!���5��#@)7l[�� �?1���5��#@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip1�0&���?!9fޥ=1Q@)��S �g�?1{<py� @:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap�0Xr�?!E�����;@)�HP�h?1�'"P�	@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��"�V?!���b�?)��"�V?1���b�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�74.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��	��R@Q�m���8@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�Ws�`�n?�Ws�`�n?!�Ws�`�n?      ��!       "	H3Mg�@H3Mg�@!H3Mg�@*      ��!       2	�;��)t�?�;��)t�?!�;��)t�?:	������@������@!������@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��	��R@y�m���8@