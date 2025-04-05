Project LexiMap: Automated Extraction and Mapping of Textbook Keywords

Abstract—This project investigates the effectiveness of automated keyword extraction using KeyBERT, a BERT-based [1] keyword extraction tool, by comparing machine-extracted keywords against the textbook's index. We evaluate the performance across different chapters using precision, recall, and F1 score metrics implemented with fuzzy matching to account for variations in keyword representation. Our initial results show promise in automating the keyword extraction process, and we plan to map the machine-extracted keywords to a knowledge graph.

INTRODUCTION
Keyword extraction is a fundamental task in natural language processing (NLP) that involves identifying the most important words or phrases that represent the main topics of a document. Recent approaches leverage transformer-based models like BERT to capture semantic meaning.

In this study, we evaluate KeyBERT, which uses BERT embeddings to extract keywords and phrases from documents. Our research question is: How well can automated keyword extraction using KeyBERT match human-curated keywords in educational texts? We use a textbook, Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow, and its index as our dataset, providing a real-world case study for educational content indexing.
METHODOLOGY
A.  Dataset
The dataset is derived from the textbook [2] containing a total of 19 chapters. Keywords are extracted and organized by chapter. We formed the dataset in this way on the basis that we will compare our dataset to the human-generated index that is also organized by chapter. By using this approach, we have a standard of comparison to evaluate our keyword extraction dataset. The training set is based on chapters 1 through 19 with only chapters 6, 10, 11, and 12 used as the test set.
B.  Keyword Extraction
We implemented a keyword extraction system using the following components: KeyBERT with a transformer model (the sentence transformer model "all-MiniLM-L6-v2"), which provides strong semantic understanding while being computationally efficient; an n-gram range of 1-3 words to capture both single terms and phrases, a diversity parameter of 0.7 to ensure variety in extracted keywords, and a maxsum algorithm to generate optimal keyword diversity.
C. Evaluation Metrics
The system's performance was evaluated with Fuzzy Matching to handle the variations in keywords between the human-generated and machine-extracted keywords. This was achieved through text normalization (lowercase and special character removal), exact match detection, substring containment evaluation, and sequence matching for fuzzy comparison. 
D. Optimization
We implemented optimizations after running into performance issues. First, a caching system was implemented to maintain consistent information across the training and the evaluating stages. This shared cache allows for significant speed improvements for repeated runs. Next, error handling was introduced to log the stages of the output; this allows for robust error recovery and result saving to prevent data loss.
E. Knowledge Graphs
At this stage in our project, the knowledge graphs have not been created yet. The conceptual model is present-- using NetworkX, take the extracted keywords and utilize the semantic understanding from KeyBERT to create edges between them.
INITIAL RESULTS
Our preliminary results show that KeyBERT can successfully extract relevant keywords from textbook chapters, but with varying degrees of success depending
Performance Metrics
We extracted keywords from the combined training chapters (1-5, 7-9, 13-19) and used these as a baseline for comparison. The average precision at this stage is 0.059 (only 5.9% of extracted keywords match ground truth); the average recall is 0.044 (only 4.4% of ground truth keywords are found); the average F1 score is 0.049 (harmonic mean is very low). On average, fewer than 3 correct keywords were found per chapter, most extracted keywords don't match ground truth, and most ground truth keywords aren't found.
Issues Identified
Thus far, we have struggled capturing the correct keywords-- the snags have been identified as following: partial matching issues cause too many terms to be captured and overly generic keywords are not filtered out.
Changes Implemented
After implementing hierarchical matching and N-gram matching, we have seen a 273% increase in average precision, a 66% increase in average recall, and a 124% increase in average F1 score.
CHALLENGES AND SOLUTIONS
Set Up 
With the Python environment and the many dependencies needed to run the ML elements, there were difficulties running the initial code.  The solution came by using Google-Colab and virtual machines. 
Collaborating
Our team needed a reliable way to share code and data while tracking changes. We established a Git repository and used GitHub for collaboration and version control. We assigned specific components to each team member based on expertise and used pull requests to review changes before merging. Regular virtual meetings helped coordinate efforts and ensure alignment on project goals.

Data Extraction
Extracting text from the textbook chapters presented several challenges. PDF extraction often resulted in formatting issues and broken text. We developed a cleaning pipeline to normalize text, remove irrelevant content like page numbers and headers, and preserve the meaningful textual content for keyword extraction. Manual verification was necessary at several points to ensure quality.

Evaluating the data
We faced challenges in developing fair evaluation metrics. The index terms didn't always perfectly match extracted keywords, requiring fuzzy matching. Additionally, choosing an appropriate similarity threshold was challenging - too low caused false positives, too high missed valid matches. We implemented hierarchical matching that considers exact matches, containment, word overlap, and string similarity to address these challenges. Continuous testing with different thresholds helped optimize the matching approach.

NEXT STEPS
Compiling results
We have gotten the keywords out of the data set but we have not yet arranged them to be of use to the mind map. In the coming weeks we will need to manipulate the results to be in a better format for our next step. 
Mind Map
After we complete data results compiling we are going to write a program to connect them with a map of the words. Each word with connections to others data points. 
Testing the results
	Our team is going to compare the data from the mind map to the index we originated the dataset from to see if we are accurate with our connections. Should we fail at this step we are going to go back to extraction and review all steps until we find the issue. 
TEAM CONTRIBUTIONS
Clark
- Led the implementation of the keyword extraction pipeline
- Developed the hierarchical matching algorithm
- Wrote data preprocessing modules and documentation
- Conducted performance analysis across different extraction parameters

Ian
- Designed and implemented the caching system for performance optimization
- Created the evaluation metrics framework with fuzzy matching
- Conducted experiments with different transformer models
- Led the documentation and report writing

Michael
- Developed the error handling and logging system
- Extracted and organized the dataset from textbook chapters
- Implemented the ground truth validation framework
- Created visualizations of results and metrics
REFERENCES
[1] M. Grootendorst, 'KeyBERT: Minimal keyword extraction with BERT'. Zenodo, 2020.
[2] A. Géron, "Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems".








