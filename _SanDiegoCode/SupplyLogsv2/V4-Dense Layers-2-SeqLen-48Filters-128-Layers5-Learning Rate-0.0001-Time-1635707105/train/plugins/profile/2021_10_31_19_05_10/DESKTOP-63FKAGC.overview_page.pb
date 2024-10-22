�	��E���@��E���@!��E���@      ��!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0��E���@���bc~?1�.Ȗ@A(���,�?I-#���y@r0*	 ��Q�nU@2T
Iterator::Root::ParallelMapV2w�T���?!����A:@)w�T���?1����A:@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeato��\���?!�@ h�9@)+~��7�?1X�/=+8@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate^�o%;�?!S�d�R9@)����N�?1�1��"�0@:Preprocessing2E
Iterator::Rootm��~���?!�'d^��B@)Ƣ��dp�?1�`@KH'@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vO~?!5���tC!@)���_vO~?15���tC!@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip����EB�?!w؛�O@)q>?�~?1&�}�!@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap�~�n�?!Vld��;@)"��u��a?1`�� @:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor8�*5{�U?!I�����?)8�*5{�U?1I�����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�70.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��V4�Q@Q��.3d=@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���bc~?���bc~?!���bc~?      ��!       "	�.Ȗ@�.Ȗ@!�.Ȗ@*      ��!       2	(���,�?(���,�?!(���,�?:	-#���y@-#���y@!-#���y@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��V4�Q@y��.3d=@�"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop�C��n��?!�C��n��?0"&
CudnnRNNCudnnRNNS�6�?!n�ݕ��?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam2��nO��?!6Y��?"C
%gradient_tape/sequential/dense/MatMulMatMul'C��?!C4����?0"(

concat_1_0ConcatV2�[�wj�?!���yny�?"C
'gradient_tape/sequential/dense/MatMul_1MatMul��~��9�?!4�JXV��?"5
sequential/dense/MatMulMatMul�7\nK�?!=��7�?0"(
gradients/AddNAddN�f[n�?!���|=��?"C
$gradients/transpose_9_grad/transpose	Transposeh�� H��?!�眂��?"*
transpose_9	Transposeh�� H��?![/��)�?Q      Y@Yn�6���@aٗ|��1W@q\�&֔S@y�7\nK�?"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�70.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�78.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 