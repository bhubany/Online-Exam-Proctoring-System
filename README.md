# Online Exam Proctoring System

## Introduction

With the shift to online education during the COVID-19 pandemic, conducting fair and secure examinations has become a challenge. The absence of proper proctoring systems has made it easier for students to engage in malpractice, degrading the quality of assessments.

The **Online Exam Proctoring System** is an AI-based solution designed to detect and prevent cheating in online exams. It utilizes:

- **Head Pose Estimation** to track head movements.
- **User Verification** to authenticate examinees.

All suspicious activities are recorded and stored as proof, ensuring a fair examination environment.

## Requirements

- **Python**: Version 3.10
- **GPU**: Recommended
- **MySQL Database**

## Setup Instructions

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/bhubany/Online-Exam-Proctoring-System.git
cd Online-Exam-Proctoring-System
```

### **Step2: Create a Virtual Environment**

```bash
python3.10 -m venv venv
```

### **Step3: Activate the Virtual Environment**

```bash
source ./venv/bin/activate
```

### **Step4: Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Step5: Configure MySQL Database**

    * Open`./oeps/settings.py`
    * Locate the *database configuration* section adn update it with your MySQL credentials

### **Step6: Migrate Database Tables**

```bash
python manage.py makemigrations
python manage.py migrate
```

### **Run the Django Server**

```bash
python manage.py runserver
```

#### **Access the Application**

    * Open browser and visit[http://127.0.0.1:8000/](http://127.0.0.1:8000/)

### Step9: Register as a User or Admin

    * User (Examinee): Takes the exam
    * Admin (Examiner): Manages and monitors exams
