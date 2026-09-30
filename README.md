\# Cervical Cell Detection System



An AI-assisted web application for classifying cervical cell images as \*\*Normal\*\* or \*\*Abnormal\*\* using a deep learning model.



The system uses a ResNet18-based image classification model trained on the Herlev Pap-Smear Dataset. It also provides probability scores and Grad-CAM visualizations to help explain which image regions contributed to the model prediction.



> \*\*Important:\*\* This project is a research/academic prototype for cervical-cell image classification. It is not a medical diagnosis system and should not be used for clinical diagnosis.



\---



\## 1. Project Objective



The objective of this project is to develop an AI-based system that can analyze cervical cell images and classify them into:



\- Normal

\- Abnormal



The project also includes an explainability component using Grad-CAM and a web interface for testing individual images.



\---



\## 2. Main Features



\- Cervical cell image upload

\- Normal/Abnormal classification

\- Prediction confidence

\- Normal and abnormal probability scores

\- Grad-CAM visualization

\- Flask REST API

\- Browser-based frontend

\- Image preprocessing using ImageNet normalization

\- ResNet18 deep learning model

\- One-click project launcher



\---



\## 3. System Workflow



```text

Cervical Cell Image

&#x20;       |

&#x20;       v

Image Upload

&#x20;       |

&#x20;       v

Image Preprocessing

&#x20;       |

&#x20;       v

ResNet18 Model

&#x20;       |

&#x20;       v

Normal / Abnormal Prediction

&#x20;       |

&#x20;       +----------------------+

&#x20;       |                      |

&#x20;       v                      v

Probability \& Confidence    Grad-CAM

&#x20;       |                      |

&#x20;       +----------+-----------+

&#x20;                  |

&#x20;                  v

&#x20;            Web Interface

