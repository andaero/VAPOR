�	���)!@���)!@!���)!@	��}x9O�?��}x9O�?!��}x9O�?"z
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails9���)!@�\5��?1[C���V@A=G仔��?I��.R(�@Y���'��?r0*	�z�GQU@2T
Iterator::Root::ParallelMapV2G�ŧ �?!Z�9};@)G�ŧ �?1Z�9};@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatg��j+��?!�i�6q;@)�m��?1����9@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��&k�C�?!6*�h57@)o��\���?1��6��)@:Preprocessing2E
Iterator::Root�R�Z��?!(0�#OD@)j�t��?1�i�6)@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��|y��?!�ǅ�$@)��|y��?1�ǅ�$@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip���b('�?!��{ܰ�M@)иp $x?1$�Z;�@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap�-�R\U�?!��ܓ9@)�p>?�`?1���͙�@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�M���PT?!͉N�kD�?)�M���PT?1͉N�kD�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 1.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�67.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��}x9O�?I�ʗ�1Q@Q�M	Rt=@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�\5��?�\5��?!�\5��?      ��!       "	[C���V@[C���V@![C���V@*      ��!       2	=G仔��?=G仔��?!=G仔��?:	��.R(�@��.R(�@!��.R(�@B      ��!       J	���'��?���'��?!���'��?R      ��!       Z	���'��?���'��?!���'��?b      ��!       JGPUY��}x9O�?b q�ʗ�1Q@y�M	Rt=@�"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop���8�n�?!���8�n�?0"&
CudnnRNNCudnnRNN�uf�a?�?!*1�/��?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam^�w(��?!�l�(���?"C
%gradient_tape/sequential/dense/MatMulMatMul^�w(��?!��ʵ�?0"C
'gradient_tape/sequential/dense/MatMul_1MatMul���5̇?!q����t�?"(

concat_1_0ConcatV2g������?!'�r)-��?"5
sequential/dense/MatMulMatMul}���?!��ѽ�.�?0"(
gradients/AddNAddNEݸ?���?!6j����?"C
$gradients/transpose_9_grad/transpose	Transpose�����8�?!<q��u��?"*
transpose_9	Transpose�����8�?!Bx�X �?Q      Y@Yn�6���@aٗ|��1W@qsP2�BiQ@y�l��k)�?"�
device�Your program is NOT input-bound because only 1.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�67.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�69.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 