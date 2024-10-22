�	|E�^k @|E�^k @!|E�^k @      ��!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0|E�^k @������?1��P�v�@Au��?I.Ȗ�[@r0*	ףp=
�Q@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat'N�w(
�?!yP,T;@)L�$zŒ?1Oc��Ә9@:Preprocessing2T
Iterator::Root::ParallelMapV27�',�?!��tA�;9@)7�',�?1��tA�;9@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���+H�?!K.��K:@)���|~�?1k΂���0@:Preprocessing2E
Iterator::RootϽ�K��?!8(3��B@)�Q,���?1���Au�(@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceO#-��#|?!ÿl��/#@)O#-��#|?1ÿl��/#@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip�,C��?!����5O@)T����p?1Fb���@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapϠ����?!�����a=@)�Q,��b?1���Au�@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�M���PT?!��n���?)�M���PT?1��n���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�71.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIB��+R@Q����mP;@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	������?������?!������?      ��!       "	��P�v�@��P�v�@!��P�v�@*      ��!       2	u��?u��?!u��?:	.Ȗ�[@.Ȗ�[@!.Ȗ�[@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qB��+R@y����mP;@�"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop�-���?!�-���?0"&
CudnnRNNCudnnRNN�D
E���?!N�g ��?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�R&� �?!~?N0��?"C
%gradient_tape/sequential/dense/MatMulMatMul 7i�!�?!Z��E�?0"(

concat_1_0ConcatV2g-� ���?!��k9u�?"5
sequential/dense/MatMulMatMulH���b�?!�w�����?0"C
'gradient_tape/sequential/dense/MatMul_1MatMul��Kߔb�?!��O0�?"(
gradients/AddNAddN��?*s�?!�����?"C
$gradients/transpose_9_grad/transpose	Transpose+�@`꤃?!΃T���?"*
transpose_9	Transpose+�@`꤃?!q��B'�?Q      Y@Yn�6���@aٗ|��1W@qՋƘT@y��J���?"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�71.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�82.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 