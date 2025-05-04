# Chest X‑Ray Diagnosis with Uncertainty Quantification

![image](https://github.com/user-attachments/assets/d0882c99-196d-4ba3-9e0e-efe306fd81a6)
figure: Web Page running on EC2 instance(output)


## Running on EC2, link to application: 
http://ec2-3-136-26-198.us-east-2.compute.amazonaws.com:5000/

This project implements a **Bayesian Convolutional Neural Network** for multi-label classification of chest X-rays (CheXpert dataset). By integrating Monte Carlo Dropout and Grad-CAM explainability, it not only predicts pathologies but also quantifies model confidence—critical for deployment in medical contexts.

## Technical Highlights

* **Base Architecture**: Fine-tuned **EfficientNet-B0** backbone using PyTorch, achieving a strong balance of accuracy and computational efficiency.
* **Uncertainty Estimation**: Employed **Monte Carlo Dropout** during inference (`T=20` stochastic passes) to compute per-class probability **means** and **variances**, flagging cases where the model is uncertain.
* **Explainability**: Generated **Grad-CAM** heatmaps on the model’s final convolutional block to visualize regions driving each prediction, aiding clinical interpretability.
* **Mixed-Precision Training**: Leveraged **\`torch.cuda.amp\`** (autocast + GradScaler) to accelerate training on NVIDIA T4 GPUs with minimal loss of numerical stability.
* **Evaluation Metrics**:

  * **Per-class ROC-AUC** tracking during validation.
  * **Calibration plots** to assess alignment between predicted confidences and empirical accuracies.

## System Design

1. **Data Pipeline**: CheXpert images loaded via Hugging Face \`datasets\`, transformed with ImageNet-style preprocessing (resize, center crop, normalize).  
2. **Model Module**: \`BayesianEfficientNet\` wraps EfficientNet-B0, adds dropout layers for Bayesian approximation, and outputs 14 pathology logits.  
3. **Inference Service**: Flask API with endpoints:

   * \`/predict\`: accepts X-ray upload, returns JSON of \`{ condition: { conf, uncertainty } }\`.  
   * Integrated with **Flask-CORS** for cross-domain AJAX requests.  
4. **Production Deployment**: Docker-style WSGI stack:

   * **Gunicorn** (4 workers) bound to \`127.0.0.1:8000\` for model serving.  
   * **Nginx** reverse proxy on port 80 for SSL termination and static file handling.  
   * Hosted on **AWS EC2** with automated systemd service for high availability.

## Demonstrated Skills

* **Deep Learning & Bayesian Methods**: Implemented uncertainty quantification in neural networks, critical for risk-sensitive applications.  
* **Model Explainability**: Applied Grad-CAM to bridge model output and human interpretation.  
* **Performance Optimization**: Mixed-precision training, dataset subsampling, and GPU utilization on EC2/T4 hardware.  
* **API Development**: Built production-grade REST service (Flask + Gunicorn + Nginx) for real-time inference.  
* **Cloud Deployment**: Configured AWS infrastructure, security groups, systemd, and reverse proxy.  
* **Software Engineering Practices**: Modular codebase, dependency management with \`requirements.txt\`, CI/CD readiness via Git workflows.

---

*This project underlines expertise in end-to-end medical image mining, Bayesian deep learning, and cloud-native deployment, showcasing capabilities essential for roles in ML engineering, deep learning research, and MLOps.*
