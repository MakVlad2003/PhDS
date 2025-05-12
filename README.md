## Overview

This repository is a personal “course‑hub” that tracks my progress through the **PhDS** (Probability & Data Science) curriculum.
Inside you will find more than **30 fully‑executed Jupyter notebooks** that move from classic statistical inference to modern deep‑learning and causal‑inference pipelines:

* Classical tests – e.g. the two‑sample **Student’s *t*‑test** for mean comparison([GitHub][1])
* Online‑experimentation methodology – **A/B‑testing** notebooks with power analysis and uplift charts([Wikipedia][2])
* Core ML algorithms – **Random Forests**([Wikipedia][3]) & **Gradient Boosting**([PyWhy][4]) for tabular data
* Fundamental deep‑learning – Multilayer Perceptrons + **Convolutional Neural Networks** (CNNs) trained on **CIFAR‑10**([Wikipedia][5])
* **Unsupervised learning** – K‑Means, hierarchical clustering & dimensionality reduction([Scikit-learn][6])
* **Causal inference** case studies using the open‑source **DoWhy** library([GitHub][7])
* Reproducible code snippets powered by SciPy/NumPy (e.g. `scipy.stats.ttest_ind`)([PyWhy][8])

The goal: a single place to revisit theory, runnable code, and accompanying slides/tasks for each topic.

---

## Repository structure

```text
PhDS/
├── HW/                       # Home‑work notebooks
│   ├── 1_Data_Analysis/
│   ├── 2_Intro_to_NN/        # feed‑forward & CIFAR‑10 CNN experiments
│   ├── 3_Intro_to_CNN/
│   ├── 4_Segmentation/
│   ├── 5_RNN/
│   ├── 6_Features_Engineering/
│   ├── 7_Random_Forest/
│   ├── 8_Gradient_Boosting/
│   ├── 9_Unsupervised_Learning/
│   ├── 10_t-tests/
│   ├── 11_AB-tests_1/
│   ├── 12_AB-tests_2/
│   └── 13_DoWhy/
├── SEM/                      # Seminar notebooks / live‑coding sessions
│   └── … (mirrors most HW topics)
├── .gitattributes            # keeps notebook *outputs* but strips broken widgets
├── .gitignore                # ignores archives & checkpoints
└── requirements.txt          # minimal runtime deps (optional, see below)
```

> **Why two trees?**
> *HW* directories hold my graded submissions; *SEM* directories hold in‑class walkthroughs and additional explorations.

---

## Quick‑start

1. **Clone** the project

   ```bash
   git clone https://github.com/MakVlad2003/PhDS.git
   cd PhDS
   ```

2. **Create an environment** (conda or venv)

   ```bash
   conda create -n phds python=3.11
   conda activate phds
   pip install -r requirements.txt   # or install packages ad‑hoc
   ```

3. **Launch JupyterLab / Notebook**

   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

4. Open any notebook and **run all cells** – all heavy‑weight datasets are either
   *downloaded on first run* (e.g. `torchvision.datasets.CIFAR10(download=True)`),
   or stored in lightweight CSV/Pickle form under `data/`.

> **Note on large files**
> Archives (`*.zip`, `*.tar.gz`, …) are Git‑ignored so the repo stays < 100 MB;
> models & figures are regenerated on‑the‑fly when you execute the notebooks.

---

## Folder‑by‑folder tour

| Folder                                  | Highlights                                             | Key concepts                                    |
| --------------------------------------- | ------------------------------------------------------ | ----------------------------------------------- |
| **1\_Data\_Analysis**                   | Pandas EDA, visual storytelling                        | descriptive stats, data cleaning                |
| **2\_Intro\_to\_NN**                    | First fully‑connected network, weight‑decay experiment | perceptron, SGD                                 |
| **3\_Intro\_to\_CNN / 4\_Segmentation** | CNN on CIFAR‑10, U‑Net mini‑lab                        | kernels, pooling, IoU metric                    |
| **5\_RNN**                              | Character‑level language model                         | back‑prop through time                          |
| **6\_Features\_Engineering**            | Target / mean encoding, WOEs                           | leakage mitigation                              |
| **7\_Random\_Forest**                   | Grid‑search with OOB score                             | bagging, feature importance                     |
| **8\_Gradient\_Boosting**               | XGBoost & CatBoost head‑to‑head                        | additive models, learning rate                  |
| **9\_Unsupervised\_Learning**           | K‑Means, Agglomerative, t‑SNE                          | elbow, silhouette                               |
| **10\_t-tests**                         | One‑ & two‑sample tests                                | Student’s *t*‑statistic([GitHub][1])            |
| **11‑12\_AB-tests**                     | Power analysis, CUPED uplift                           | A/B experimentation([Wikipedia][2])             |
| **13\_DoWhy**                           | “Effect of ads on conversions” causal graph            | back‑door, propensity score, DoWhy([GitHub][7]) |

---

## Re‑encoding & widget stripping

Broken Jupyter‑widget metadata sometimes prevents GitHub from rendering notebooks.
A custom **`nbsoft`** Git filter (see `.gitattributes`) keeps legitimate *outputs* yet
removes the offending `metadata.widgets` key so notebooks stay viewable online.

If you add new notebooks:

```bash
git add <my_notebook>.ipynb      # the filter runs automatically
git commit -m "Add new lab"
```

---

## Contributing

Pull‑requests are welcome!
Feel free to open an issue for bugs, suggestions or additional exercises.

---

## License

All notebooks & code are released under the **MIT License**.
Datasets retain their original licenses; see individual notebook headers for attributions.

---

### Acknowledgements

*Scikit‑learn, PyTorch, CatBoost, DoWhy and the wider open‑source community* – made learning & experimentation friction‑free.
Icons courtesy of the GitHub Octicons set.

---

Enjoy exploring – and happy modelling! 🚀

---

<!-- CITATIONS -->

The README references:

* Student’s *t*‑test([GitHub][1])
* CNNs
* Random Forests([Wikipedia][3])
* Gradient Boosting([PyWhy][4])
* Unsupervised learning([Scikit-learn][6])
* A/B testing([Wikipedia][2])
* CIFAR‑10([Wikipedia][5])
* SciPy `ttest_ind` API([PyWhy][8])
* DoWhy library([GitHub][7])

[1]: https://github.com/MakVlad2003/PhDS/tree/main/HW "PhDS/HW at main · MakVlad2003/PhDS · GitHub"
[2]: https://en.wikipedia.org/wiki/A/B_testing "A/B testing - Wikipedia"
[3]: https://en.wikipedia.org/wiki/Student%27s_t-test "Student's t-test - Wikipedia"
[4]: https://pywhy.github.io/dowhy/latest/ "Internal Error"
[5]: https://en.wikipedia.org/wiki/CIFAR-10 "CIFAR-10 - Wikipedia"
[6]: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html "Internal Error"
[7]: https://github.com/py-why/dowhy "GitHub - py-why/dowhy: DoWhy is a Python library for causal inference that supports explicit modeling and testing of causal assumptions. DoWhy is based on a unified language for causal inference, combining causal graphical models and potential outcomes frameworks."
[8]: https://pywhy.github.io/dowhy/latest/getting_started.html "Internal Error"
