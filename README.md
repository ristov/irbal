## IRBAL - Informativeness Ranking-Based Active Learning

This repository contains the implementation of IRBAL used for NIDS alert classification on the NAD-CD dataset (https://github.com/ristov/nad-cd). The implementation requires the Python scikit-learn, pandas, modAL-python, and catboost packages. To display all command line options and their descriptions, run:

python3 irbal.py -h

The implementation of IRBAL with concept drift detector additionally requires the Python river package. To display all command line options and their descriptions, run:

python3 irbal-with-detector.py -h

**Author:** Risto Vaarandi (firstname d0t lastname at gmail d0t c0m)
