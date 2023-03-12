{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c85883b4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# config.py\n",
    "ALL_FIELDS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',\n",
    "             'marital-status', 'occupation', 'relationship', 'race',\n",
    "             'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'country']\n",
    "CONT_FIELDS = ['age', 'fnlwgt', 'education-num',\n",
    "               'capital-gain', 'capital-loss', 'hours-per-week']\n",
    "CAT_FIELDS = list(set(ALL_FIELDS).difference(CONT_FIELDS))\n",
    "\n",
    "# Hyper-parameters for Experiment\n",
    "NUM_BIN = 10\n",
    "BATCH_SIZE = 256\n",
    "EMBEDDING_SIZE = 5"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
