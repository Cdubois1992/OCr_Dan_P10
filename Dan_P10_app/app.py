## Important : 
# Le lancement de cette application requiert 2 arguments : 
# Le premier > Le Jeu de donnée a utiliser
# Le second > Le seuil de tolérance de la prédiction.
#               - Doit être une valeur comprise entre 0 et 1. 
#               - 0.95 signfifie que la probabilité d'attribuer le type 'Vrai billet' doit être de 95% ou plus. 
#               - Valeurs conseillées : entre 0.60 et 1

# Importation des libraries necessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.impute import SimpleImputer 
from sklearn.linear_model import LogisticRegression
import sys
import os


# Importation des modèles ajustés (fitted models)

Imputer_fitted=pickle.load(open('Imputer_fitted.pkl', 'rb'))
logisticRegrT_fitted=pickle.load(open('logisticRegrT_fitted.pkl', 'rb'))
Knn_fitted=pickle.load(open('Knn_fitted.pkl', 'rb'))



# Création d'un dossier d'output

os.makedirs('Output',exist_ok=True) 
os.makedirs('Output/LogisticRegression',exist_ok=True) 
os.makedirs('Output/KNearestN',exist_ok=True) 

# Importation du jeu de donnés

billets = pd.read_csv(sys.argv[1],sep=',')


# Tableau des informations du jeu de donnée
Table_infos = []

for col in billets.columns:
    CountVal = billets[col].count()
    Nullval  = billets[col].isnull().sum()
    ColType = str(billets[col].dtype)
    Table_infos.append([col, CountVal, Nullval,ColType])

Table_infos = pd.DataFrame(Table_infos)   
Table_infos.columns = ['Variables','Nb_billets','Billets_sans_infos' ,'type_variable']

# Filtre de la colonne id

billets_F= billets.drop(columns = ['id'])


# Histogramme des données

plt.rcParams["figure.figsize"] = (16,8)
fig, axes = plt.subplots(2,3)
fig.suptitle('Histogramme des données')

sns.set(font_scale=1.5)
sns.histplot(ax=axes[0,0],data=billets_F, x="diagonal")
sns.histplot(ax=axes[0,1],data=billets_F, x="height_left")
sns.histplot(ax=axes[0,2],data=billets_F, x="height_right")
sns.histplot(ax=axes[1,0],data=billets_F, x="margin_low")
sns.histplot(ax=axes[1,1],data=billets_F, x="margin_up")
sns.histplot(ax=axes[1,2],data=billets_F, x="length")

plt.savefig('Output/histogramme.png', dpi=300)


# Informations statistiques des données 

infosstat = round(billets_F.describe(),3)


# Imputation des données manquantes

billets_Imp = pd.DataFrame(Imputer_fitted.transform(billets_F), columns=billets_F.columns)

# Prédictions par Regression Logistique


pred_RL = logisticRegrT_fitted.predict(billets_Imp)
pred_RL = np.where(pred_RL == 1, 'Vrai', 'Faux')
pred_RL = pd.Series(pred_RL)

Proba_vrai_RL = logisticRegrT_fitted.predict_proba(billets_Imp)
Proba_vrai_RL = pd.Series(np.round(Proba_vrai_RL[:,1],2))


# Prédictions par KNearest Neighbors

pred_KNN = Knn_fitted.predict(billets_Imp)
pred_KNN = np.where(pred_KNN == True, 'Vrai', 'Faux')
pred_KNN = pd.Series(pred_KNN)


Proba_vraiknn = Knn_fitted.predict_proba(billets_Imp)
Proba_vraiknn = pd.Series(np.round(Proba_vraiknn[:,1],2))



# Génération du tableau des résultats

billets_output = pd.concat([billets,
                            pred_RL.rename('Type_billet_RL'),
                            Proba_vrai_RL.rename('Prob_Vrai_RL'),
                            pred_KNN.rename('Type_billet_KNN'),
                            Proba_vraiknn.rename('Prob_Vrai_KNN') ], axis=1)


billets_output.to_csv('Output/billets_output.csv')

# Création d'un tableau comprenant les estimations peu fiables
Seuil = float(sys.argv[2])

Billets_seuilRL = billets_output.query('1-@Seuil <= Prob_Vrai_RL <= @Seuil')
Billets_seuilRL.to_csv('Output/LogisticRegression/Billets_seuilRL.csv')


Billets_seuilKNN = billets_output.query('1-@Seuil <= Prob_Vrai_KNN <= @Seuil')
Billets_seuilKNN.to_csv('Output/KNearestN/Billets_seuilKNN.csv')

billets_output_FiltreSeuilRL = billets_output.drop(Billets_seuilRL.index, axis=0)
billets_output_FiltreSeuilRL.to_csv('Output/LogisticRegression/billets_output_FiltreSeuilRL.csv')

billets_output_FiltreSeuilKNN = billets_output.drop(Billets_seuilKNN.index, axis=0)
billets_output_FiltreSeuilKNN.to_csv('Output/KNearestN/billets_output_FiltreSeuilKNN.csv')

# Création d'un tableau comptant les faux billets (sur le seuil)

Faux_billets_sum = []
FB_sum_RL = (billets_output.Type_billet_RL == 'Faux').sum()
FB_sum_KNN = (billets_output.Type_billet_KNN == 'Faux').sum()
Faux_billets_sum.append([FB_sum_RL, FB_sum_KNN])
Faux_billets_sum = pd.DataFrame(Faux_billets_sum)   
Faux_billets_sum.columns = ['Faux_billets_RL','Faux_billets_KNN']




# Représentation graphisue des résultats de la regression Logistique 

fig, axs = plt.subplots(ncols=3,figsize=(20,6))

p1=sns.scatterplot(data = billets_output, x='length', y= 'margin_up', hue='Type_billet_RL',ax=axs[0],legend = False)
p1.set( xlabel = "Longueur(mm)", ylabel = "Marge sup. (mm)")

p2=sns.scatterplot(data = billets_output, x='length', y= 'margin_low', hue='Type_billet_RL',ax=axs[1],legend = False)
p2.set( xlabel = "Longueur(mm)", ylabel = "Marge inf. (mm)")

p3=sns.scatterplot(data = billets_output, x='margin_low', y='margin_up', hue='Type_billet_RL',ax=axs[2])
p3.set( xlabel = "Marge inf. (mm)", ylabel = "Marge sup. (mm)")

plt.savefig('Output/LogisticRegression/fig2.png')


# Représentation graphisue des résultats de la méthode KNN 

fig, axs = plt.subplots(ncols=3,figsize=(20,6))

p4=sns.scatterplot(data = billets_output, x='length', y= 'margin_up', hue='Type_billet_KNN',ax=axs[0],legend = False)
p4.set( xlabel = "Longueur(mm)", ylabel = "Marge sup. (mm)")

p5=sns.scatterplot(data = billets_output, x='length', y= 'margin_low', hue='Type_billet_KNN',ax=axs[1],legend = False)
p5.set( xlabel = "Longueur(mm)", ylabel = "Marge inf. (mm)")

p6=sns.scatterplot(data = billets_output, x='margin_low', y='margin_up', hue='Type_billet_KNN',ax=axs[2])
p6.set( xlabel = "Marge inf. (mm)", ylabel = "Marge sup. (mm)")

plt.savefig('Output/KNearestN/fig3.png')





# Création du rapport HTML en 'string'
html = f'''
    <html>
        <head>
            <title > Rapport de l'analyse des faux billets </title>
        </head>
        <body>
            <h1>Rapport de l'analyse des faux billets </h1>
            <h2> I) Description des données</h2>
            <h3> I-1) Résumé des données</h3>
            <p> Le tableau ci-dessous présente la liste des variables ainsi que leur type, le nombre de billets et le nombre de billets comportant des informations manquantes.</p>
            {Table_infos.to_html()}
            <h3> I-2) Descriptions statistiques </h3>
            <p> Le graphe ci-dessous présente l'histogramme de chaque variable </p>
            <img src='histogramme.png' width="1600" align="center">
            <p> Tableau du résumé statistique des données </p>
            {infosstat.to_html()}
            <h2> II) Prédictions des Vrais et Faux billets </h2>
            <p> Cette application de détection utilise deux algorithmes performants de prédiction : La régression Logistique et la méthode des K Nearest Neighbors.
            <br> Les modèles utilisés ont été entraîné et testé au préalable sur un jeu de données comportant 1000 vrais billets et 500 faux billets. </p>
            <h3>II-1) Tableau des résultats</h3>
            {billets_output.head().to_html()}
            <p> Le nombre de faux billets detectés par chacune des méthode de l'algorithme est détaillé dans le tableau ci-dessous </p>
            {Faux_billets_sum.to_html()}
            <h3>II-2) Réprésentation graphique des résultats</h3>
            <p> La différence entre les vrais et les faux billets s'expliquent principalement par la marge sup, la marge inf et la longueur du billet.
            <br>Ces différentes variables sont représentées ci-dessous avec les valeurs de chaque billets, ainsi que la prédiction de leur type.  </p>
            <h4> a) Prédictions par Régression Logistique </h4> 
            <img src='LogisticRegression/fig2.png' width="1600" align="center">
            <h4> b) Prédictions par KNN </h4> 
            <img src='KNearestN/fig3.png' width="1600" align="center">
         </body>
    </html>
    '''
# conversion du html string en html
with open('Output/html_report.html', 'w') as f:
    f.write(html)
  

# Print du résumé dans la console
print("L'algorithme a detecté",Faux_billets_sum['Faux_billets_RL'].values[0],"faux billets par Régression Logistique et",Faux_billets_sum['Faux_billets_KNN'].values[0],"faux billets par la méthode des K Nearest Neighbors. Le rapport et les tableaux des prédictions ont été générés.")
