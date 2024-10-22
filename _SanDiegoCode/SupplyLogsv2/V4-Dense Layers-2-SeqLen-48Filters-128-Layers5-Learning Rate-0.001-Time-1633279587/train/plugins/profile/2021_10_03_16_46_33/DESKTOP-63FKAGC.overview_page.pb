�	z�"n�.@z�"n�.@!z�"n�.@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-z�"n�.@�<Fy�e@1����?A,,����?IS�Z��@*	�����9S@2U
Iterator::Model::ParallelMapV2�0�*�?!����:@)�0�*�?1����:@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateM�O��?!��ΓD:@)�q����?1�o��<I4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatvq�-�?!�`�dы4@)_�Qڋ?1S�/p�1@:Preprocessing2F
Iterator::Model���H�?!�t��D@)Ǻ����?1/r�� -@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapݵ�|г�?!�ĳ�Q@@)n��t?1�C4c�|@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceHP�s�r?!˝&�[�@)HP�s�r?1˝&�[�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip,e�X�?!�&�^�RM@)�J�4q?1��·�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!��r�	�@)/n��b?1��r�	�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 50.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�36.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIj�3_�U@Q��"g�(@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�<Fy�e@�<Fy�e@!�<Fy�e@      ��!       "	����?����?!����?*      ��!       2	,,����?,,����?!,,����?:	S�Z��@S�Z��@!S�Z��@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qj�3_�U@y��"g�(@�"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop6MV��$�?!6MV��$�?0"&
CudnnRNNCudnnRNN#�W����?!,�4��?"(

concat_1_0ConcatV2����R�?!8r�p��?"*
transpose_9	Transpose������?!#I ��?"(
gradients/AddNAddN��P�#ǃ?!U��-�?"C
$gradients/transpose_9_grad/transpose	Transpose�
��ƃ?!J}�N8|�?";
gradients/split_2_grad/concatConcatV2�(��ڭ�?!�
����?"-
IteratorGetNext/_8_Recv��8��z?!��z���?"C
'gradient_tape/sequential/dense/MatMul_1MatMul�W��^z?!�����0�?"S
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad�W��^z?!W$�T�e�?Q      Y@Y�M��dG@a';r��W@q�'�UͣS@y���#x�?"�
both�Your program is POTENTIALLY input-bound because 50.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�36.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�78.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 