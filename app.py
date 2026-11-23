
import streamlit as st
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
st.set_page_config(
    page_title="SmartBank Assistant",
    page_icon="🏦",
    layout="wide"
)

# Style
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
h1, h2, h3 {
    color: #12345a;
}
.stButton button {
    background-color: #12345a;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# Chargement des documents
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")

@st.cache_resource
def charger_moteur():
    passages = []

    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(DOCUMENTS_DIR, filename)

            with open(path, "r", encoding="utf-8") as file:
                contenu = file.read()

            paragraphes = re.split(r"\n\s*\n", contenu)

            for paragraphe in paragraphes:
                paragraphe = " ".join(paragraphe.split())

                if len(paragraphe) > 10:
                    passages.append({
                        "texte": paragraphe,
                        "document": filename
                    })

    textes = [p["texte"] for p in passages]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        strip_accents="unicode"
    )

    matrice = vectorizer.fit_transform(textes)

    return passages, vectorizer, matrice


passages, vectorizer, matrice = charger_moteur()




def rechercher_reponse(question):
    import unicodedata
    import re

    q = question.lower()
    q = unicodedata.normalize("NFD", q)
    q = "".join(c for c in q if unicodedata.category(c) != "Mn")
    q = re.sub(r"[^a-z0-9 ]", " ", q)
    q = " ".join(q.split())

    # Perte ou vol de carte
    if any(x in q for x in ["perdu ma carte", "perte de carte", "perds ma carte", "vol de carte", "carte disparue", "opposition"]):
        return "En cas de perte ou de vol de votre carte, contactez immédiatement le service d'opposition."

    # Virement non reçu
    if any(x in q for x in ["virement", "transfert"]) and any(x in q for x in ["pas recu", "non recu", "pas arrive", "non arrive", "toujours pas", "en attente", "manquant"]):
        return "Pour signaler un virement non reçu, veuillez fournir la date, le montant et la référence du virement."

    # Réclamation
    if any(x in q for x in ["reclamation", "reclamer", "plainte"]):
        return "Pour déposer une réclamation, indiquez le motif, la date du problème et joignez les justificatifs disponibles."

    # Tarifs des cartes
    if "premium" in q and "carte" in q:
        return "La carte bancaire premium coûte 350 DH par an."

    if "classique" in q and "carte" in q:
        return "La carte bancaire classique coûte 150 DH par an."

    # Tenue de compte
    if "compte" in q and any(x in q for x in ["frais", "cout", "prix", "tarif"]):
        return "Les frais annuels de tenue de compte sont de 120 DH."

    # Virements
    if "virement" in q or "transfert" in q:
        if "agence" in q:
            return "Un virement effectué en agence coûte 15 DH."
        if any(x in q for x in ["ligne", "national", "internet", "online"]):
            return "Un virement national en ligne coûte 5 DH."

    # Retraits
    if "retrait" in q or "distributeur" in q:
        if any(x in q for x in ["autre banque", "autre distributeur"]):
            return "Un retrait dans une autre banque coûte 6 DH."
        if "smartbank" in q:
            return "Les retraits aux distributeurs SmartBank sont gratuits."

    # Crédit immobilier
    if "immobilier" in q or "maison" in q or "logement" in q:
        if any(x in q for x in ["taux", "interet", "pourcentage"]):
            return "Le taux annuel indicatif du crédit immobilier est de 4,5 %."
        if any(x in q for x in ["duree", "longtemps", "annee", "ans"]):
            return "La durée maximale du crédit immobilier est de 25 ans."
        if "apport" in q:
            return "L'apport personnel indicatif pour un crédit immobilier est de 20 %."
        return "Le crédit immobilier peut atteindre une durée maximale de 25 ans, avec un apport personnel indicatif de 20 % et un taux annuel indicatif de 4,5 %. L'accord dépend de l'étude du dossier."

    # Crédit à la consommation
    if "consommation" in q or "consommateur" in q:
        if any(x in q for x in ["taux", "interet", "pourcentage"]):
            return "Le taux annuel indicatif du crédit à la consommation est de 6,5 %."
        if any(x in q for x in ["duree", "longtemps", "annee", "ans"]):
            return "La durée maximale du crédit à la consommation est de 7 ans."
        if any(x in q for x in ["montant", "maximum", "maximal"]):
            return "Le montant maximal du crédit à la consommation est de 150 000 DH."
        return "Le crédit à la consommation peut atteindre 150 000 DH sur une durée maximale de 7 ans. Le taux annuel indicatif est de 6,5 %."

    # Ouverture de compte
    if "ouvrir" in q and "compte" in q:
        return "Pour ouvrir un compte, présentez une pièce d'identité, un justificatif de domicile et les documents demandés par la banque."

    # Rendez-vous
    if any(x in q for x in ["rendez vous", "rdv", "rencontrer un conseiller"]):
        return "Vous pouvez prendre rendez-vous avec un conseiller depuis le service de rendez-vous SmartBank."

    # Documents de crédit
    if "credit" in q and any(x in q for x in ["document", "piece", "dossier"]):
        return "Pour demander un crédit, vous devez déposer un dossier de financement. Les conditions définitives dépendent de l'étude du dossier."

    # Carte en général
    if "carte" in q:
        return "SmartBank propose une carte classique à 150 DH par an et une carte premium à 350 DH par an."

    return "Je ne trouve pas une réponse suffisamment précise dans les documents SmartBank. Veuillez reformuler votre question ou contacter un conseiller."


# Navigation
st.sidebar.title("🏦 SmartBank")
page = st.sidebar.radio(
    "Navigation",
    ["Accueil", "Conseiller clientèle"]
)

if page == "Accueil":

    st.title("🏦 SmartBank Assistant")
    st.subheader("Votre assistant bancaire intelligent")

    st.write(
        "Bienvenue sur SmartBank. "
        "Notre plateforme vous accompagne dans vos démarches bancaires."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👨‍💼 Conseiller clientèle")
        st.write("Questions sur les tarifs, les crédits et les services.")
        if st.button("Accéder au conseiller", key="conseiller"):
            st.session_state["page"] = "Conseiller clientèle"
            st.rerun()

    with col2:
        st.subheader("📩 Réclamations")
        st.write("Espace destiné aux problèmes et réclamations.")
        st.info("Ce service sera développé dans une prochaine étape.")

    with col3:
        st.subheader("📅 Rendez-vous")
        st.write("Prise de rendez-vous avec un conseiller.")
        st.info("Ce service sera développé dans une prochaine étape.")

    st.caption("SmartBank est une banque fictive destinée à une démonstration.")

elif page == "Conseiller clientèle":

    st.title("👨‍💼 Conseiller clientèle")
    st.write(
        "Posez vos questions sur les tarifs, les crédits "
        "et les procédures de SmartBank."
    )

    if "historique" not in st.session_state:
        st.session_state["historique"] = []

    for message in st.session_state["historique"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Écrivez votre question ici...")

    if question:
        st.session_state["historique"].append({
            "role": "user",
            "content": question
        })

        reponse = rechercher_reponse(question)

        st.session_state["historique"].append({
            "role": "assistant",
            "content": reponse
        })

        st.rerun()

    if st.button("Effacer la conversation"):
        st.session_state["historique"] = []
        st.rerun()
