# import streamlit as st
# from PIL import Image
# with open("style.css") as f:
#     st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
# #####################
# # Header 
# st.write('''
# # Syed Abdul Qadir Gilani
# ##### *Resume* 
# ''')
# image = Image.open('dp.png')
# st.image(image, width=200)
# st.markdown('## Summary', unsafe_allow_html=True)
# st.info('''
# - Experienced Educator, Researcher and Administrator with almost twenty years of experience in data-oriented environment and a passion for delivering insights based on predictive modeling. 
# - Strong verbal and written communication skills as demonstrated by extensive participation as invited speaker at `10` conferences as well as publishing 149 research articles.
# - Strong track record in scholarly research with H-index of `32` and total citation of 3200+.
# ''')

# #####################
# # Navigation

# st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

# st.markdown("""
# <nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #16A2CB;">
#   <a class="navbar-brand" href="https://youtube.com/dataprofessor" target="_blank">Chanin Nantasenamat</a>
#   <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
#     <span class="navbar-toggler-icon"></span>
#   </button>
#   <div class="collapse navbar-collapse" id="navbarNav">
#     <ul class="navbar-nav">
#       <li class="nav-item active">
#         <a class="nav-link disabled" href="/">Home <span class="sr-only">(current)</span></a>
#       </li>
#       <li class="nav-item">
#         <a class="nav-link" href="#education">Education</a>
#       </li>
#       <li class="nav-item">
#         <a class="nav-link" href="#work-experience">Work Experience</a>
#       </li>
#       <li class="nav-item">
#         <a class="nav-link" href="#bioinformatics-tools">Bioinformatics Tools</a>
#       </li>
#       <li class="nav-item">
#         <a class="nav-link" href="#social-media">Social Media</a>
#       </li>
#     </ul>
#   </div>
# </nav>
# """, unsafe_allow_html=True)

# #####################
# # Custom function for printing text
# def txt(a, b):
#   col1, col2 = st.columns([4,1])
#   with col1:
#     st.markdown(a)
#   with col2:
#     st.markdown(b)

# def txt2(a, b):
#   col1, col2 = st.columns([1,4])
#   with col1:
#     st.markdown(f'`{a}`')
#   with col2:
#     st.markdown(b)

# def txt3(a, b):
#   col1, col2 = st.columns([1,2])
#   with col1:
#     st.markdown(a)
#   with col2:
#     st.markdown(b)
  
# def txt4(a, b, c):
#   col1, col2, col3 = st.columns([1.5,2,2])
#   with col1:
#     st.markdown(f'`{a}`')
#   with col2:
#     st.markdown(b)
#   with col3:
#     st.markdown(c)

# #####################
# st.markdown('''
# ## Education
# ''')

# txt('**Doctor of Philosophy** (Medical Technology), *Mahidol University*, Thailand',
# '2002-2006')
# st.markdown('''
# - GPA: `3.89`
# - Research thesis entitled `Computer-aided molecular design for biological and chemical applications : Quantum chemical and machine learning approach`.
# - Received Royal Golden Jubilee Ph.D. Scholarship of `2.152 million THB` covering tuition and stipend.
# - Thesis awarded `1st` Prize by the National Research Council of Thailand.
# ''')

# txt('**Bachelors of Science** (Biological Science), *Mahidol University International College*, Thailand',
# '1998-2002')
# st.markdown('''
# - GPA: `3.65`
# - Graduated with First Class Honors.
# ''')

# #####################
# st.markdown('''
# ## Work Experience
# ''')

# txt('**Head, Center of Data Mining and Biomedical Informatics**, Faculty of Medical Technology, Mahidol University, Thailand',
# '2011-2021')
# st.markdown('''
# - Managing a Center of `10` professors, researchers and students to ensure KPIs are strategically achieved namely to publish at least `20+` research publications annually. 
# - Actively took part in the talent hiring process as well as help employees to plan and develop their career path.
# - Set budget and handle procurement in order to facilitate education and research activities. Secured `> 10 million THB` budget.
# - Set and reflect on OKR on an annual basis to ensure productivity strategically matches the organization's direction.
# ''')

# txt('**Associate Professor**, Faculty of Medical Technology, Mahidol University, Thailand',
# '2012-2021')
# txt('**Assistant Professor**, Faculty of Medical Technology, Mahidol University, Thailand',
# '2009-2012')
# txt('**Lecturer**, Faculty of Medical Technology, Mahidol University, Thailand',
# '2006-2009')
# st.markdown('''
# - Provided mentorship and supervision to junior faculty, researchers, Ph.D./M.Sc./B.Sc. students. Mentored `3` Post-doctoral fellows, supervised `13` Ph.D. students, supervised `8` M.Sc. students, supervised `13` B.Sc. students and hosted `6` visiting students from U.S., Sweden and India.
# - Wrote and applied for research grants. Served as Principal Investigator for research grants that have been awarded `12.5 million THB` and `1.117 million SEK` in research funding from Thai and Swedish grant agencies.
# - Conducted research by applying machine learning to computational drug discovery and ensuring that research output exceeds `20+` articles per year.
# - Taught `10+` undergraduate/graduate classes on Bioinformatics, Data Mining, Scientific Research and Presentation, Research Methodology, Graduate Seminar, Programming for Health Data Science, etc.
# - Peer reviewed `100+` research articles for leading scientific journals.
# ''')

# txt('**Co-Chair**, International Conference on Pharmaceutical Bioinformatics at Pattaya, Thailand',
# '2016')
# st.markdown('''
# - Oversee all aspects of the conference preparations from conception to launch. This include inviting keynote and plenary speakers, create advertisement flyers, create abstract book, etc.
# - Conference attracted `200+` participants from `40+` countries from university and industry sector.
# - Chaired keynote session, technical workshop and some of the parallel sessions.
# ''')

# txt('**Content Creator**, [Data Professor YouTube Channel](https://youtube.com/dataprofessor/)',
# '2019-Present')
# st.markdown('''
# - `100,000+` subscribers on YouTube
# - Created `261` educational videos on data science, machine learning and bioinformatics as well as hosted several podcast episodes with data scientists.
# - Created `3` sponsored videos for Notion, Gradio and Classpert.
# ''')

# txt('**Content Creator**, [Coding Professor YouTube Channel](https://youtube.com/codingprofessor/)',
# '2019-Present')
# st.markdown('''
# - `3,200+` subscribers on YouTube
# - Created `38` educational videos on Python and R programming.
# ''')

# txt('**Technical Writer**, [Data Professor Blog](https://data-professor.medium.com/) on Medium.com',
# '2019-Present')
# st.markdown('''
# - `4,100+` subscribers on Medium
# - Written `68` technical blogs on data science, machine learning and bioinformatics.
# ''')

# #####################
# st.markdown('''
# ## Bioinformatics Tools
# ''')
# txt4('ABCpred', 'A web server for the discovery of acetyl- and butyryl-cholinesterase inhibitors', 'http://codes.bio/abcpred/')
# txt4('AutoWeka', 'An automated data mining software based on Weka', 'http://www.mt.mahidol.ac.th/autoweka/')
# txt4('ACPred', 'A computational tool for the prediction and analysis of anticancer peptides','http://codes.bio/acpred/')
# txt4('BioCurator', 'A web server for curating ChEMBL bioactivity data', 'http://codes.bio/biocurator/')
# txt4('CryoProtect', 'A web server for classifying antifreeze proteins from non-antifreeze proteins','http://codes.bio/cryoprotect/')
# txt4('ERpred', 'A web server for the prediction of subtype-specific estrogen receptor antagonists', 'http://codes.bio/erpred')
# txt4('HCVpred', 'A web server for predicting the bioactivity of Hepatitis C virus NS5B inhibitors', 'http://codes.bio/hemopred/')
# txt4('HemoPred', 'A web server for predicting the hemolytic activity of peptides', 'http://codes.bio/hemopred/')
# txt4('iQSP', 'A sequence-based tool for the prediction and analysis of quorum sensing peptides', 'http://codes.bio/iqsp/')
# txt4('Meta-iAVP', 'A sequence-based meta-predictor for improving the prediction of antiviral peptides', 'http://codes.bio/meta-iavp/')
# txt4('osFP', 'A web server for predicting the oligomeric state of fluorescent proteins', 'http://codes.bio/osfp/')
# txt4('PAAP', 'A web server for predicting antihypertensive activity of peptides', 'http://codes.bio/paap/')
# txt4('PepBio', 'A web server for predicting the bioactivity of host defense peptide', 'http://codes.bio/pepbio')
# txt4('PyBact', 'Open source software written in Python for bacterial identification', 'https://sourceforge.net/projects/pybact/')
# txt4('TargetAntiAngio', 'A sequence-based tool for the prediction and analysis of anti-angiogenic peptides','http://codes.bio/targetantiangio/')
# txt4('ThalPred', 'Development of decision model for discriminating Thalassemia trait and Iron deficiency anemia','http://codes.bio/thalpred/')
# txt4('THPep', 'A web server for predicting tumor homing peptides','http://codes.bio/thpep/')


# #####################
# st.markdown('''
# ## Skills
# ''')
# txt3('Programming', '`Python`, `R`, `Linux`')
# txt3('Data processing/wrangling', '`SQL`, `pandas`, `numpy`')
# txt3('Data visualization', '`matplotlib`, `seaborn`, `plotly`, `altair`, `ggplot2`
# txt3('Machine Learning', '`scikit-learn`')
# txt3('Deep Learning', '`TensorFlow`')
# txt3('Web development', '`HTML`, `CSS`, `JavaScript`, `Flask`')
# txt3('Model deployment', '`streamlit`, `AWS`')

# #####################
# st.markdown('''
# ## Social Media
# ''')
# txt2('LinkedIn', 'https://www.linkedin.com/in/syedabdulqadir/')
# txt2('Twitter', 'https://twitter.com/SyedAbdulQadir_')
# txt2('GitHub', 'https://github.com/SyedAbdulQadirGilani001 ')
# # import argparse
# # import cv2
# # import numpy as np
# # import matplotlib.pyplot as plt
# # ap = argparse.ArgumentParser()
# # ap.add_argument("-i", "--image", required=True, help="Path to the image")
# # args = vars(ap.parse_args())
# # image = cv2.imread(args["image"])
# # cv2.imshow("Original", image)
# # chans = cv2.split(image)
# # colors = ("b", "g", "r")
# # plt.figure()
# # plt.title("'Flattened' Color Histogram")
# # plt.xlabel("Bins")
# # plt.ylabel("# of Pixels")
# # features = []
# # for (chan, color) in zip(chans, colors):
# #     hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
# #     features.extend(hist)
# #     plt.plot(hist, color=color)
# #     plt.xlim([0, 256])
# # plt.show()
import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
st.write("""
# Simple Iris Flower Prediction App

This app predicts the  **Iris flower** type!

""")
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine dataset"))
classifier_name = st.sidebar.selectbox("Select Classifier", ("SVM", "Random Forest"))
def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y
X, y = get_dataset(dataset_name)  
st.write("Shape of dataset", X.shape)
st.write("Number of classes", len(np.unique(y)))
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params
params = add_parameter_ui(classifier_name)
def get_classifier(clf_name, params):
    if clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
        max_depth=params["max_depth"], random_state=1234)
    return clf
clf = get_classifier(classifier_name, params)
# Classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = ", acc)
# Plot
pca = PCA(2)
X_projected = pca.fit_transform(X)
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]
fig = plt.figure()
plt.scatter(x1, x2,
c=y, alpha=0.8,
cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
# plt.show()
st.pyplot(fig)
