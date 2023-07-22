# **Investigating women’s employment status tendency among Canadian couples and families**

## **[Keyword]**

Socioeconomics, Statistical Modeling/Analysis, Generalized Linear Model (GLM), Exploratory Data Analysis, Data Visualization, R Programming

## **[Overview]**

I analyzed the tendency of women’s employment status concerning their **marital status**, **income of the male member in the household**, **presence of children**, and **region of residence**. The dataset was based on 1977 surveys of Canadian couples and families. The model was **GLM** in the binomial family since the dataset had both continuous and categorical variables.

## **[Approach]**

- **Exploratory data analysis** (EDA) by observing the correlations, 
  - between the outcome variable (i.e., women’s employment status) and input variables and 
  - among the input variables
- **Model-fitting in various scenarios**: 
  - using all input variables, 
  - using fewer variables, 
  - using all input variables one of which interacts with other input variables, and 
  - using fewer variables and one of which interacts with remaining input variables
- **Evaluation on models** based on the p-value of **ANOVA** test, **AIC** score, and **AUROC** score

## **[Outcome]**

Based on the analysis, the implications were,

- the strong association of the **presence of children** and 
- the slight association of **male members’ income** 

to women’s employment status. I **achieved high distinction** for this task.