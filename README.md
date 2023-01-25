# Class-independent Regularization for Learning with Noisy Labels
PyTorch Code for the following paper at AAAI2023:\
<b>Title</b>: <i>Class-independent Regularization for Learning with Noisy Labels</i>\
<b>Authors</b>:Rumeng Yi, Dayan Guan, Yaping Huang, Shijian Lu


<b>Abstract</b>\
Training deep neural networks (DNNs) with noisy labels often leads to poorly generalized models as DNNs tend to memorize the noisy labels in training. Various strategies have been developed for improving sample selection precision and mitigating the noisy label memorization issue. However, most existing works adopt a class-dependent softmax classifier that is vulnerable to noisy labels by entangling the classification of multi-class features. This paper presents a class-independent regularization (CIR) method that can effectively alleviate the negative impact of noisy labels in DNN training. CIR regularizes the class-dependent softmax classifier by introducing multi-binary classifiers each of which takes care of one class only. Thanks to its class-independent nature, CIR is tolerant to noisy labels as misclassification by one binary classifier does not affect others. For effective training of CIR, we design a heterogeneous adaptive co-teaching strategy that forces the class-independent and class-dependent classifiers to focus on sample selection and image classification, respectively, in a cooperative manner. Extensive experiments show that CIR achieves superior performance consistently across multiple benchmarks with both synthetic and real images.
