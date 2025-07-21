# HR Analytics Challenge - Previsão de Attrition

Projeto final da disciplina **Data Science Experience** (Mackenzie) com foco em **Machine Learning Aplicado a RH**. O objetivo foi desenvolver um sistema preditivo para identificar funcionários com alto risco de deixar a empresa, com base em dados sintéticos inspirados no IBM HR Analytics.

---

## 🔍 Objetivo do Projeto

A TechCorp Brasil enfrenta uma taxa de rotatividade de 35%, com perdas estimadas em R$ 45 milhões. Este projeto entrega:

- Pipeline completo de Machine Learning
- Engenharia de atributos com 15+ features criadas
- Avaliação com 4 algoritmos
- Técnicas de balanceamento (SMOTE)
- Interpretação com SHAP
- Relatório técnico e visualizações

---

## 🧠 Tecnologias Utilizadas

- `Python 3.10+`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`, `xgboost`, `shap`
- `imblearn`, `joblib`, `PolynomialFeatures`

---

## 📁 Estrutura do Repositório
projeto-ml-mack-py/
├── data/
│ └── processed/ # Dados tratados (.parquet)
├── models/ # Modelos e scalers salvos
├── projeto_melhorado.ipynb # Notebook principal
├── requirements.txt # Bibliotecas necessárias
└── README.md # Este documento

---

## 📈 Resultados

- **Precision-Recall AUC:** até 0.81 com XGBoost
- **F1-Score:** até 0.73 (modelo balanceado)
- **SHAP:** insights interpretáveis sobre risco de saída

---

## 📌 Recomendações de Negócio

- Monitoramento de risco por departamento
- Ações corretivas em equilíbrio trabalho-vida
- Retenção focada em funcionários júnior e em viagem

---

## ▶️ Como Executar

1. Clone o repositório:
```bash
git clone https://github.com/Arthurperes/projeto-ml-mack-py.git
cd projeto-ml-mack-py

2. Instale as dependências:
pip install -r requirements.txt

3. Rode o notebook:
jupyter notebook projeto.ipynb
