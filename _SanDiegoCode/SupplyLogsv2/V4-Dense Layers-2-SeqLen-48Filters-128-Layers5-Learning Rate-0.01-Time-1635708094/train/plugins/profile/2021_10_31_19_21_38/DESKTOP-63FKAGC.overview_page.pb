�	�r��#@�r��#@!�r��#@	��St���?��St���?!��St���?"z
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails9�r��#@�^����?1|E��@AP�c*�?IJ�({K�@Yj����?r0*	������Y@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatqU�wE�?![�п>@)Gx$(�?1]���3f<@:Preprocessing2T
Iterator::Root::ParallelMapV2Qf�L2r�?!'�DM#5@)Qf�L2r�?1'�DM#5@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateW!�'�>�?!	���}�;@)�g?RD��?1�'ZE4@:Preprocessing2E
Iterator::Root���G�Ƞ?!��"D�?@)�-���=�?1_���$@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip�ߡ(�'�?!�\��Q@)e�X��?1� �Ǯ @:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��VC�~?!&��2z@)��VC�~?1&��2z@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap�=~oӟ?!�;���=@)Mۿ�Ҥd?1�-�W�p@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�"��\?!�og����?)�"��\?1�og����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 6.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�69.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��St���?I�
��S@QWd�d�6@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�^����?�^����?!�^����?      ��!       "	|E��@|E��@!|E��@*      ��!       2	P�c*�?P�c*�?!P�c*�?:	J�({K�@J�({K�@!J�({K�@B      ��!       J	j����?j����?!j����?R      ��!       Z	j����?j����?!j����?b      ��!       JGPUY��St���?b q�
��S@yWd�d�6@�"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop���n��?!���n��?0"&
CudnnRNNCudnnRNN
�I-���?!�H4���?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam_{n���?!ټ�D��?"C
%gradient_tape/sequential/dense/MatMulMatMulb����?!��Qw	�?0"C
'gradient_tape/sequential/dense/MatMul_1MatMulU.�9C�?!���^j�?"(

concat_1_0ConcatV2*o�?!<-�s��?"(
gradients/AddNAddNR�2UT�?!���o�'�?"5
sequential/dense/MatMulMatMulR�2UT�?!6�o���?0"C
$gradients/transpose_9_grad/transpose	TransposeE��q�?!C�6�x��?"*
transpose_9	TransposeE��q�?!P���!�?Q      Y@Yn�6���@aٗ|��1W@q��g �O@yR�2UT�?"�
both�Your program is POTENTIALLY input-bound because 6.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�69.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�63.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 