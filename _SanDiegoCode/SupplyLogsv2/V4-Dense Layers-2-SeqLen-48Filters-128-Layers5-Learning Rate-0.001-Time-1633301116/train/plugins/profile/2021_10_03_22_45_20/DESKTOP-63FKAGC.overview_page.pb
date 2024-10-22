�	��-�"@��-�"@!��-�"@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��-�"@&���#�?1+j0� @A���Xǵ?I%!���@*	33333�U@2U
Iterator::Model::ParallelMapV2��_vO�?!�T^~j9@)��_vO�?1�T^~j9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatQ�|a2�?!�f�p�7@)	�^)ː?1Wf��� 3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate{�G�z�?!&���,7@)���S㥋?1��EI/@:Preprocessing2F
Iterator::Modelz6�>W[�?!̰ԭ
�C@)�(��0�?1���U�,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��<,Ԫ?!4O+R�[N@)��~j�t�?1��('&@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice9��v��z?!~�8_� @)9��v��z?1~�8_� @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor"��u��q?!0��3�@)"��u��q?10��3�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapA��ǘ��?!�H6Wf�9@)/n��b?1�Ni��d@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 13.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�62.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIh�o��JS@Qa�AJ��6@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	&���#�?&���#�?!&���#�?      ��!       "	+j0� @+j0� @!+j0� @*      ��!       2	���Xǵ?���Xǵ?!���Xǵ?:	%!���@%!���@!%!���@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qh�o��JS@ya�AJ��6@�"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop�/�����?!�/�����?0"&
CudnnRNNCudnnRNNN���˷�?!���ZJ�?"(

concat_1_0ConcatV2�"�H>�?!&}��S��?"(
gradients/AddNAddN�i_%�^�?!���Y��?"C
$gradients/transpose_9_grad/transpose	Transpose�4�'<Z�?!��1J7n�?"*
transpose_9	Transpose��ACR�?!H9W���?";
gradients/split_2_grad/concatConcatV2�eK&N�?!��f��?""
split_1Split�)��jz?!�`xҎA�?"-
IteratorGetNext/_7_Sendg,�W�y?!��ځ�t�?"C
%gradient_tape/sequential/dense/MatMulMatMul���.}Zv?!d{8|R��?0Q      Y@Y�3[
@a��N^W@q)\6�,�O@yLo��?"�
both�Your program is POTENTIALLY input-bound because 13.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�62.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�63.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 