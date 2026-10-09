# Diabetes Project

Your first machine learning project. You formulate hypotheses about a real medical dataset, explore and clean it to check them, and build a logistic regression that predicts whether a patient has diabetes. You decide which evaluation metric matters most, beat a simple baseline, and make the model catch more diabetes cases by handling the class imbalance.

## Learning Objectives

By the end of this repository, you should be able to:

- Choose an evaluation metric for a medical screening problem, and justify it by which mistake costs more: a missed case or a false alarm.
- Formulate a research question and three hypotheses before exploring the data, test each one on the training set with a statistical test, and give it a verdict.
- Find the missing values **hidden as zeros**, and impute them with statistics from the training set only.
- Build a random and a rule-based baseline, and compare them with a logistic regression on your chosen metric.
- Handle the class imbalance with `class_weight="balanced"` and with a lower decision threshold chosen on the training data.
- Report the training and test scores of every model in one table, and read the confusion matrix of the best model to explain where it still makes mistakes.

## Learning Path

> [!TIP]
> Start with the [**Diabetes Project**](01_diabetes_project.ipynb) notebook. Its task section takes you through every step of the project, from choosing a metric to evaluating all models on the test set, and the diagram at the top shows how the steps fit together.

| File / Folder | Description |
|---|---|
| [**01 - Diabetes Project**](01_diabetes_project.ipynb) | The project: explore and clean the data, then build, compare and evaluate logistic regression models that predict diabetes. |

### Additional Folders and Files

| File / Folder | Description |
|---|---|
| [**Data**](data/) | `diabetes.csv`, the Pima Indians Diabetes dataset, one row per patient with `Outcome` as the target. |
| [**Assets**](assets/) | The diagram of the project's steps, with its Mermaid source in `assets/diagrams/`. |
| [**Paper on the Diabetes Mellitus Data Set**](paper_on_diabetes_mellitus_data_set.pdf) | Background paper on the dataset. |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock) | Dependency lock file. |

## Setup

> [!NOTE]
> Throughout these steps, text in angle brackets like `<repo-name>` is a **placeholder**. Replace it including the `< >` brackets with your own value. For example, `cd <repo-name>` becomes `cd my-diabetes-project`.

### 1. Create the Repository from the Template

Click **Use this template** on GitHub.

When creating the repository:

- Set yourself as the **Owner**
- Choose a repository name
- Disable **Include all branches**
- Click **Create repository**

> [!IMPORTANT]
> If you are working in pairs or groups, only **one person** should complete this step.

---

### 2. Add Collaborators (Pairs/Groups Only)

If working with teammates:

1. Open the repository on GitHub
2. Go to **Settings → Collaborators**
3. Add your teammates as collaborators
4. Share the repository link with your team

Teammates should accept the invitation before continuing.

---

### 3. Clone the Repository

Copy the SSH URL from the **Code** button on GitHub, then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL will look like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the Project Folder and Install Dependencies

This installs all dependencies and creates a virtual environment in (`.venv/`).

```bash
cd <repo-name>
uv sync
```

---


### 5. Open the Notebook

> [!NOTE]
> Make sure you open VS Code from the project root so it automatically detects the environment created by `uv sync`.

Launch VS Code in the project root folder:

```bash
code .
```

Then open the notebook and select the Python environment created by `uv sync` as the kernel.

## References & Further Reading

- [**Pima Indians Diabetes Database**](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database): The original dataset on Kaggle, with column descriptions.
- [**scikit-learn: LogisticRegression**](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html): The model you use, including the `class_weight` parameter and `predict_proba`.
- [**scikit-learn: Precision, recall and F-measures**](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics): The metrics you choose between, and how they are calculated.
- [**scikit-learn: Precision-Recall**](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html): How moving the decision threshold trades precision for recall.
- [**8 Tactics to Combat Imbalanced Classes**](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/): Background on why class imbalance matters and the ways to handle it.
