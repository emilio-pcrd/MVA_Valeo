# MVA Valeo

Quality control in industrial settings plays a crucial role in ensuring product reliability while preventing defective components from propagating through production lines. Traditionally, this process
relies on human inspection, where images of manufactured parts are analyzed to detect potential defects. However, manual inspections are prone to human error and can represent additional workload,
making them inefficient for large-scale industrial applications. Deep learning modeling has recently
made significant strides in the world of computer vision, greatly extending foundation models to industrial applications. This technologies promise good control over image classification tasks. Image
classification systems rely on image inputs being classified as a class or a label.
In this project, as part of a challenge proposed by Valeo, we integrate deep learning-based computer
vision models into the quality control workflow. The aim is to classify product images into known
defect categories while also detecting anomalous images that may not fit predefined defect classes.

To achieve this, we explore advanced machine learning techniques, including self-supervised learning for anomaly detection (PaDiM with SimCLR), CNN classification (ResNet101), and confidence-based filtering to enhance decision-making reliability.


The report is organized as follow: after presenting the `problem framework` and describing the dataset,
highlighting challenges such as **class imbalance** and **unlabeled anomalies**, we introduce the baseline
model provided by the challenge organizers, and then expose our proposed improvements and final
methodology. In the last section, we discuss our **experimental results**, `benchmark different architectures` and analyze their implications.
