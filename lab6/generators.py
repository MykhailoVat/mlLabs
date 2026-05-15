import numpy as np
from sklearn.model_selection import train_test_split

# CONFIG
NUM_CAMPAIGNS = 500
APPLICANTS_PER_CAMPAIGN = 1500
TOTAL_SEATS = 350
PRIVILEGED_MAX = int(TOTAL_SEATS * 0.10)

# BUSINESS-LOGIC
def compute_score(exams_res):
    m, u, e = exams_res
    return m * 0.4 + u * 0.3 + e * 0.3

def is_eligible_non_priv(exams_res, score):
    m, u, e = exams_res
    return (m >= 140) and (score >= 160)

def is_eligible_priv(exams_res, score):
    m, u, e = exams_res
    return (m >= 120) and (u >= 120) and (e >= 120) and (score >= 144)

def assign_admission(exams_ress, privs, scores):
    length = len(exams_ress)

    eligible_idx = []

    for i in range(length):
        if privs[i] == 1:
            if is_eligible_priv(exams_ress[i], scores[i]):
                eligible_idx.append(i)
        else:
            if is_eligible_non_priv(exams_ress[i], scores[i]):
                eligible_idx.append(i)

    eligible_idx = sorted(eligible_idx, key=lambda i: scores[i], reverse=True)

    assigned = np.zeros(length)

    priv_count = 0
    total_count = 0

    for i in eligible_idx:
        if privs[i] == 1:
           if total_count >= TOTAL_SEATS:
               break
           if priv_count >= PRIVILEGED_MAX:
               break

           assigned[i] = 1
           priv_count += 1
           total_count += 1
        else:
            if total_count >= TOTAL_SEATS:
                break

            assigned[i] = 1
            total_count += 1

    return assigned


# ONE-HOT
def make_one_hot(campaign_id):
    v = np.zeros(NUM_CAMPAIGNS)
    v[campaign_id] = 1
    return v

# GENERATORS
def generate_student_ability(peak):
    ability = np.random.normal(loc=peak,scale=10)
    return np.clip(ability,100,200)


def generate_exams_res(ability):
    noise = np.array([
        np.random.normal(0, 7),
        np.random.normal(0, 7),
        np.random.normal(0, 7)
    ])

    res = ability + noise

    res = np.clip(
        np.round(res),100,200
    )

    return res.astype(int)

def generate_campaign():
    exams_ress, privs = [], []
    scores = []

    camp_peak = np.random.randint(170, 185)
    p_priv = np.random.uniform(0.05, 0.20)

    for _ in range(APPLICANTS_PER_CAMPAIGN):
        ability = generate_student_ability(camp_peak)
        exams_res = generate_exams_res(ability)
        priv = np.random.rand() < p_priv
        score = compute_score(exams_res)

        exams_ress.append(exams_res)
        privs.append(int(priv))
        scores.append(score)

    exams_ress = np.array(exams_ress)
    privs = np.array(privs)
    scores = np.array(scores)

    return exams_ress, privs, scores


def generate_dataset(campaigns = NUM_CAMPAIGNS):
    features_all = []
    assigns_all = []

    for cid in range(campaigns):
        exams_ress, privs, scores = generate_campaign()
        y = assign_admission(exams_ress, privs, scores)

        mean = np.mean(scores)
        #print(mean)
        #std = np.std(scores)

        campaign_stats = np.tile(
            [mean],
            (APPLICANTS_PER_CAMPAIGN, 1)
        )

        features = np.hstack([
            exams_ress,
            privs.reshape(-1, 1),
            campaign_stats
        ])

        features_all.append(features)
        assigns_all.append(y)


    return (
        np.vstack(features_all),
        np.hstack(assigns_all),
    )


def generate_dataset_split(test_size=0.2, random_state=42):
    features, assigns = generate_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        assigns,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=assigns
    )

    return X_train, X_test, y_train, y_test