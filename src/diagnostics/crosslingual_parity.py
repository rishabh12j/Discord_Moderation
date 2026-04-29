"""
Change 4 — Cross-Lingual Parity Diagnostic.

Tests whether semantically equivalent content in different languages
gets the same moderation decision and similar toxicity scores.

10 test sentences spanning the full severity spectrum, manually translated
into all 13 languages. Run BEFORE calibration to establish baseline disparity,
then AFTER to measure improvement.

RUN:
  python -m src.diagnostics.crosslingual_parity
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from collections import defaultdict

# ── Test sentences: 10 severity levels × 13 languages ───────────
# Each dict maps language code → translation
# Translations aim for semantic equivalence, not word-for-word

TEST_SENTENCES = [
    {
        "id": "benign_greeting",
        "severity": "benign",
        "expected_action": "ALLOW",
        "translations": {
            "en": "Good morning everyone, hope you're having a great day!",
            "ru": "Доброе утро всем, надеюсь у вас отличный день!",
            "uk": "Доброго ранку всім, сподіваюсь у вас чудовий день!",
            "de": "Guten Morgen zusammen, ich hoffe ihr habt einen tollen Tag!",
            "es": "Buenos días a todos, espero que estén teniendo un gran día!",
            "ar": "صباح الخير للجميع، أتمنى أن يومكم رائع!",
            "hi": "सभी को सुप्रभात, उम्मीद है आपका दिन अच्छा हो!",
            "zh": "大家早上好，希望你们今天过得愉快！",
            "it": "Buongiorno a tutti, spero che stiate passando una bella giornata!",
            "fr": "Bonjour à tous, j'espère que vous passez une bonne journée!",
            "he": "בוקר טוב לכולם, מקווה שיש לכם יום נהדר!",
            "ja": "皆さんおはようございます、良い一日をお過ごしください！",
            "hin": "Sabko good morning, hope kar raha hoon aapka din achha ho!",
        },
    },
    {
        "id": "benign_gameplay",
        "severity": "benign",
        "expected_action": "ALLOW",
        "translations": {
            "en": "Nice play on that last round, really well done!",
            "ru": "Отличная игра в последнем раунде, молодец!",
            "uk": "Гарна гра в останньому раунді, молодець!",
            "de": "Toller Spielzug in der letzten Runde, wirklich gut gemacht!",
            "es": "Buena jugada en la última ronda, muy bien hecho!",
            "ar": "لعبة رائعة في الجولة الأخيرة، أحسنت!",
            "hi": "पिछले राउंड में बढ़िया खेल, बहुत अच्छा किया!",
            "zh": "上一轮打得很好，真的做得不错！",
            "it": "Bella giocata nell'ultimo round, davvero ben fatto!",
            "fr": "Beau jeu au dernier tour, vraiment bien joué!",
            "he": "משחק יפה בסיבוב האחרון, באמת כל הכבוד!",
            "ja": "前のラウンドのプレイ素晴らしかった、お見事！",
            "hin": "Last round mein bahut achha khela, really well done!",
        },
    },
    {
        "id": "mild_frustration",
        "severity": "mild",
        "expected_action": "ALLOW/WARN",
        "translations": {
            "en": "This team is so bad, can anyone even play this game?",
            "ru": "Эта команда такая плохая, кто-нибудь вообще умеет играть?",
            "uk": "Ця команда така погана, хтось взагалі вміє грати?",
            "de": "Dieses Team ist so schlecht, kann überhaupt jemand spielen?",
            "es": "Este equipo es tan malo, ¿alguien sabe jugar?",
            "ar": "هذا الفريق سيء جداً، هل يعرف أحد اللعب أصلاً؟",
            "hi": "ये टीम इतनी खराब है, क्या कोई खेलना भी जानता है?",
            "zh": "这个队太差了，有人会玩这个游戏吗？",
            "it": "Questa squadra è così scarsa, qualcuno sa giocare?",
            "fr": "Cette équipe est tellement nulle, est-ce que quelqu'un sait jouer?",
            "he": "הקבוצה הזאת כל כך גרועה, מישהו בכלל יודע לשחק?",
            "ja": "このチームひどすぎる、誰かこのゲームできるの？",
            "hin": "Ye team itni bad hai, koi khel bhi sakta hai kya?",
        },
    },
    {
        "id": "direct_insult",
        "severity": "moderate",
        "expected_action": "WARN/DELETE",
        "translations": {
            "en": "You are all garbage players, absolutely useless.",
            "ru": "Вы все мусорные игроки, абсолютно бесполезные.",
            "uk": "Ви всі сміттєві гравці, абсолютно марні.",
            "de": "Ihr seid alle Müllspieler, absolut nutzlos.",
            "es": "Todos ustedes son jugadores basura, completamente inútiles.",
            "ar": "أنتم جميعاً لاعبون زبالة، عديمو الفائدة تماماً.",
            "hi": "तुम सब कूड़ा खिलाड़ी हो, बिल्कुल बेकार।",
            "zh": "你们都是垃圾玩家，完全没用。",
            "it": "Siete tutti giocatori spazzatura, completamente inutili.",
            "fr": "Vous êtes tous des joueurs poubelle, complètement inutiles.",
            "he": "כולכם שחקני זבל, חסרי תועלת לחלוטין.",
            "ja": "お前ら全員ゴミプレイヤーだ、全く役に立たない。",
            "hin": "Tum sab kachra players ho, bilkul bekaar.",
        },
    },
    {
        "id": "harsh_insult",
        "severity": "high",
        "expected_action": "DELETE",
        "translations": {
            "en": "Shut up you braindead idiots, uninstall the game.",
            "ru": "Заткнитесь, безмозглые идиоты, удалите игру.",
            "uk": "Заткніться, безмозкі ідіоти, видаліть гру.",
            "de": "Haltet die Fresse ihr hirntoten Idioten, deinstalliert das Spiel.",
            "es": "Cállense idiotas sin cerebro, desinstalen el juego.",
            "ar": "اخرسوا أيها البلهاء عديمو العقل، احذفوا اللعبة.",
            "hi": "चुप रहो बेवकूफों, गेम अनइंस्टॉल करो।",
            "zh": "闭嘴你们这些白痴，卸载游戏吧。",
            "it": "State zitti idioti decerebrati, disinstallate il gioco.",
            "fr": "Fermez-la bande d'idiots sans cervelle, désinstallez le jeu.",
            "he": "שתקו אידיוטים חסרי מוח, תמחקו את המשחק.",
            "ja": "黙れ脳なしのバカども、ゲームをアンインストールしろ。",
            "hin": "Chup raho bewakoofo, game uninstall karo.",
        },
    },
    {
        "id": "severe_toxic",
        "severity": "extreme",
        "expected_action": "DELETE/TIMEOUT",
        "translations": {
            "en": "You worthless pieces of trash, I hope you all suffer.",
            "ru": "Вы никчёмные куски мусора, надеюсь вы все пострадаете.",
            "uk": "Ви нікчемні шматки сміття, сподіваюсь ви всі постраждаєте.",
            "de": "Ihr wertlosen Stücke Müll, ich hoffe ihr leidet alle.",
            "es": "Pedazos de basura inútiles, espero que todos sufran.",
            "ar": "أنتم قطع قمامة لا قيمة لها، أتمنى أن تعانوا جميعاً.",
            "hi": "तुम सब बेकार कचरे के टुकड़े हो, मुझे उम्मीद है तुम सब दुख भोगो।",
            "zh": "你们这些毫无价值的垃圾，我希望你们都受苦。",
            "it": "Pezzi di spazzatura senza valore, spero che soffriate tutti.",
            "fr": "Espèces de déchets sans valeur, j'espère que vous souffrirez tous.",
            "he": "חתיכות זבל חסרי ערך, אני מקווה שכולכם תסבלו.",
            "ja": "お前ら価値のないゴミクズだ、全員苦しめばいい。",
            "hin": "Tum sab bekaar kachra ke tukde ho, mujhe umeed hai tum sab dukh bhogo.",
        },
    },
    {
        "id": "threat_veiled",
        "severity": "threat",
        "expected_action": "TIMEOUT/BAN",
        "translations": {
            "en": "I will find where you live and make you regret this.",
            "ru": "Я найду где ты живёшь и заставлю тебя пожалеть.",
            "uk": "Я знайду де ти живеш і змушу тебе пошкодувати.",
            "de": "Ich werde herausfinden wo du wohnst und es dich bereuen lassen.",
            "es": "Voy a encontrar dónde vives y te arrepentirás de esto.",
            "ar": "سأجد أين تعيش وسأجعلك تندم على هذا.",
            "hi": "मैं पता लगाऊंगा तुम कहाँ रहते हो और तुम्हें पछतावा करवाऊंगा।",
            "zh": "我会找到你住在哪里，让你后悔。",
            "it": "Troverò dove vivi e te ne pentirai.",
            "fr": "Je trouverai où tu habites et je te ferai regretter.",
            "he": "אני אמצא איפה אתה גר ואגרום לך להתחרט.",
            "ja": "お前の住所を突き止めて後悔させてやる。",
            "hin": "Main pata lagaunga tum kahan rehte ho aur tumhe pachhataawa karwaunga.",
        },
    },
    {
        "id": "hate_speech",
        "severity": "extreme",
        "expected_action": "DELETE/TIMEOUT",
        "translations": {
            "en": "People like you don't deserve to exist, you're subhuman.",
            "ru": "Такие как ты не заслуживают существования, ты недочеловек.",
            "uk": "Такі як ти не заслуговують існування, ти недолюдина.",
            "de": "Leute wie du verdienen es nicht zu existieren, du bist Untermensch.",
            "es": "Gente como tú no merece existir, eres infrahumano.",
            "ar": "أشخاص مثلك لا يستحقون الوجود، أنت أقل من إنسان.",
            "hi": "तुम जैसे लोग जीने के लायक नहीं हैं, तुम इंसान से कम हो।",
            "zh": "像你这样的人不配活着，你不算人。",
            "it": "Gente come te non merita di esistere, sei un subumano.",
            "fr": "Les gens comme toi ne méritent pas d'exister, tu es sous-humain.",
            "he": "אנשים כמוך לא ראויים להתקיים, אתה תת-אנושי.",
            "ja": "お前みたいな奴は存在する価値がない、人間以下だ。",
            "hin": "Tum jaise log jeene ke laayak nahi hain, tum insaan se kam ho.",
        },
    },
    {
        "id": "passive_aggressive",
        "severity": "subtle",
        "expected_action": "ALLOW/WARN",
        "translations": {
            "en": "Wow, that was quite a play. Really impressive decision making there.",
            "ru": "Ого, вот это была игра. Действительно впечатляющие решения.",
            "uk": "Ого, оце була гра. Дійсно вражаючі рішення.",
            "de": "Wow, das war ja ein Spielzug. Wirklich beeindruckende Entscheidungen.",
            "es": "Vaya, esa fue toda una jugada. Decisiones realmente impresionantes.",
            "ar": "واو، كانت هذه لعبة رائعة. قرارات مثيرة للإعجاب حقاً.",
            "hi": "वाह, क्या खेल था। सच में प्रभावशाली फैसले।",
            "zh": "哇，这操作真是绝了。决策真是令人印象深刻啊。",
            "it": "Wow, che bella giocata. Decisioni davvero impressionanti.",
            "fr": "Wow, quel jeu. Des décisions vraiment impressionnantes.",
            "he": "וואו, איזה משחק. החלטות באמת מרשימות.",
            "ja": "わー、すごいプレイだったね。本当に素晴らしい判断だ。",
            "hin": "Wow, kya play tha. Really impressive decision making.",
        },
    },
    {
        "id": "death_threat",
        "severity": "extreme_threat",
        "expected_action": "BAN",
        "translations": {
            "en": "I swear I will kill you, you're dead.",
            "ru": "Клянусь, я тебя убью, ты труп.",
            "uk": "Клянуся, я тебе вб'ю, ти труп.",
            "de": "Ich schwöre ich bringe dich um, du bist tot.",
            "es": "Juro que te mataré, estás muerto.",
            "ar": "أقسم أنني سأقتلك، أنت ميت.",
            "hi": "कसम खाता हूँ मैं तुझे मार डालूंगा, तू मरा हुआ है।",
            "zh": "我发誓我会杀了你，你死定了。",
            "it": "Giuro che ti ammazzo, sei morto.",
            "fr": "Je jure que je vais te tuer, tu es mort.",
            "he": "אני נשבע שאני אהרוג אותך, אתה מת.",
            "ja": "誓って殺してやる、お前は死んだ。",
            "hin": "Kasam khaata hoon main tujhe maar dalunga, tu mara hua hai.",
        },
    },
]

LANGUAGES = ["en", "ru", "uk", "de", "es", "ar", "hi", "zh", "it", "fr", "he", "ja", "hin"]
LANG_NAMES = {
    "en": "English", "ru": "Russian", "uk": "Ukrainian", "de": "German",
    "es": "Spanish", "ar": "Arabic", "hi": "Hindi", "zh": "Chinese",
    "it": "Italian", "fr": "French", "he": "Hebrew", "ja": "Japanese",
    "hin": "Hinglish",
}


def run_parity_test():
    """
    Phase 1: Raw toxicity score comparison.
    Uses the XLM-R classifier directly — no model needed.
    Shows per-language score distributions for each test sentence.
    """
    from transformers import pipeline as hf_pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    judge = hf_pipeline(
        "text-classification",
        model="textdetox/xlmr-large-toxicity-classifier-v2",
        device=device,
        truncation=True,
        max_length=512,
    )

    def score_text(text):
        if not text.strip():
            return 0.0
        result = judge(text)[0]
        is_toxic = result["label"].lower() in ["toxic", "1", "label_1"]
        return result["score"] if is_toxic else 1.0 - result["score"]

    print("=" * 80)
    print("CROSS-LINGUAL PARITY DIAGNOSTIC — RAW CLASSIFIER SCORES")
    print("=" * 80)
    print()

    all_scores = defaultdict(list)  # lang → [scores across all sentences]
    per_sentence = []               # for detailed report

    for sent in TEST_SENTENCES:
        sent_id = sent["id"]
        severity = sent["severity"]
        scores = {}

        for lang in LANGUAGES:
            text = sent["translations"][lang]
            scores[lang] = score_text(text)
            all_scores[lang].append(scores[lang])

        # Compute spread
        score_vals = list(scores.values())
        mean_score = np.mean(score_vals)
        std_score = np.std(score_vals)
        min_lang = min(scores, key=scores.get)
        max_lang = max(scores, key=scores.get)
        spread = scores[max_lang] - scores[min_lang]

        per_sentence.append({
            "id": sent_id,
            "severity": severity,
            "scores": scores,
            "mean": mean_score,
            "std": std_score,
            "spread": spread,
            "min_lang": min_lang,
            "max_lang": max_lang,
        })

        # Print per-sentence results
        flag = "" if spread > 0.30 else "   "
        print(f"{flag}[{severity:>14s}] {sent_id}")
        print(f"     Mean={mean_score:.3f}  Std={std_score:.3f}  "
              f"Spread={spread:.3f}  (lowest: {min_lang}={scores[min_lang]:.3f}, "
              f"highest: {max_lang}={scores[max_lang]:.3f})")

        # Show all 13 scores in a compact row
        score_strs = [f"{lang}={scores[lang]:.3f}" for lang in LANGUAGES]
        # Print in two rows of ~6-7 to fit
        print(f"     {', '.join(score_strs[:7])}")
        print(f"     {', '.join(score_strs[7:])}")
        print()

    # ── Per-language summary ────────────────────────────────────
    print("=" * 80)
    print("PER-LANGUAGE AVERAGE SCORES (across all test sentences)")
    print("=" * 80)

    en_mean = np.mean(all_scores["en"])
    print(f"\n   {'Language':12s} {'Mean':>8s} {'Std':>8s} {'vs English':>10s}")
    print(f"   {'─'*42}")
    for lang in LANGUAGES:
        lmean = np.mean(all_scores[lang])
        lstd = np.std(all_scores[lang])
        diff = lmean - en_mean
        flag = "" if abs(diff) > 0.10 else "  "
        print(f"   {LANG_NAMES[lang]:12s} {lmean:8.3f} {lstd:8.3f} {diff:+10.3f} {flag}")

    # ── Disparity analysis ──────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("DISPARITY ANALYSIS")
    print("=" * 80)

    # Which sentences have worst cross-lingual parity?
    worst = sorted(per_sentence, key=lambda x: x["spread"], reverse=True)
    print(f"\n  Sentences with worst cross-lingual spread (> 0.30 = problematic):")
    for s in worst[:5]:
        icon = "" if s["spread"] > 0.30 else "" if s["spread"] > 0.15 else ""
        print(f"    {icon} {s['id']:25s} spread={s['spread']:.3f}  "
              f"(low: {s['min_lang']}={s['scores'][s['min_lang']]:.3f}, "
              f"high: {s['max_lang']}={s['scores'][s['max_lang']]:.3f})")

    # Which languages are consistently under/over-scored?
    print(f"\n  Per-language bias (negative = under-scored vs English):")
    biases = []
    for lang in LANGUAGES:
        if lang == "en":
            continue
        # Compare each sentence's score to English score
        diffs = [all_scores[lang][i] - all_scores["en"][i]
                 for i in range(len(TEST_SENTENCES))]
        avg_diff = np.mean(diffs)
        biases.append((lang, avg_diff))

    biases.sort(key=lambda x: x[1])
    for lang, bias in biases:
        icon = "" if bias < -0.15 else "" if bias < -0.05 else ""
        direction = "UNDER-scored" if bias < -0.05 else "OVER-scored" if bias > 0.05 else "parity"
        print(f"    {icon} {LANG_NAMES[lang]:12s}: avg bias = {bias:+.3f} ({direction})")

    # Actionable verdict
    print(f"\n  VERDICT:")
    max_spread = max(s["spread"] for s in per_sentence)
    avg_spread = np.mean([s["spread"] for s in per_sentence])
    severely_biased = [l for l, b in biases if abs(b) > 0.15]

    if avg_spread < 0.10 and not severely_biased:
        print(f"    Cross-lingual parity is good (avg spread={avg_spread:.3f})")
    elif avg_spread < 0.20:
        print(f"    Moderate disparity (avg spread={avg_spread:.3f})")
        print(f"       Per-language calibration recommended.")
    else:
        print(f"    Severe disparity (avg spread={avg_spread:.3f}, max={max_spread:.3f})")
        print(f"       Per-language calibration REQUIRED before production use.")
        if severely_biased:
            print(f"       Most affected: {', '.join(LANG_NAMES[l] for l in severely_biased)}")


def run_threat_detection_test():
    """
    Phase 1.5: Test multilingual threat detection (Change 3).
    No model needed — tests regex patterns directly.
    """
    try:
        from src.agent.production_moderator import detect_threat
    except ImportError:
        print("\nCannot import detect_threat — skipping threat detection test")
        return

    print(f"\n{'=' * 80}")
    print("MULTILINGUAL THREAT DETECTION COVERAGE (Change 3)")
    print("=" * 80)

    threat_sentences = [s for s in TEST_SENTENCES if s["severity"] in ("threat", "extreme_threat")]

    total_hits = 0
    total_tests = 0

    for sent in threat_sentences:
        print(f"\n  [{sent['severity']}] {sent['id']}:")
        hits = []
        misses = []

        for lang in LANGUAGES:
            text = sent["translations"][lang]
            is_threat, category = detect_threat(text)
            total_tests += 1
            if is_threat:
                hits.append((lang, category))
                total_hits += 1
            else:
                misses.append(lang)

        print(f"     Detected ({len(hits)}/{len(LANGUAGES)}): "
              f"{', '.join(f'{LANG_NAMES[l]}[{c}]' for l, c in hits)}")
        if misses:
            print(f"     MISSED: {', '.join(LANG_NAMES[l] for l in misses)}")
        else:
            print(f"     Perfect coverage!")

    # Also test that benign sentences don't trigger false positives
    print(f"\n  False positive check (benign sentences):")
    benign_sentences = [s for s in TEST_SENTENCES if s["severity"] == "benign"]
    fp_count = 0
    for sent in benign_sentences:
        for lang in LANGUAGES:
            text = sent["translations"][lang]
            is_threat, category = detect_threat(text)
            if is_threat:
                fp_count += 1
                print(f"     FALSE POSITIVE: {sent['id']} in {LANG_NAMES[lang]} → {category}")
    if fp_count == 0:
        print(f"     No false positives on benign content")

    coverage = total_hits / max(total_tests, 1) * 100
    print(f"\n  THREAT DETECTION COVERAGE: {total_hits}/{total_tests} ({coverage:.0f}%)")


def run_production_parity_test():
    """
    Phase 2: Full production moderator comparison.
    Tests whether the same semantic content gets the same DECISION across languages.
    Now also reports threat detection hits from Change 3.
    Requires trained model.
    """
    try:
        from src.agent.production_moderator import ProductionModerator
    except Exception as e:
        print(f"\nCannot load production moderator: {e}")
        print(f"    Skipping Phase 2. Run Phase 1 results above for baseline.")
        return

    print(f"\n\n{'=' * 80}")
    print("CROSS-LINGUAL PARITY — PRODUCTION DECISIONS")
    print("=" * 80)

    moderator = ProductionModerator()

    consistency_results = []

    for sent in TEST_SENTENCES:
        sent_id = sent["id"]
        severity = sent["severity"]
        expected = sent["expected_action"]
        decisions = {}
        tox_scores = {}
        threats = {}

        for lang in LANGUAGES:
            text = sent["translations"][lang]
            user_id = f"parity_test_{lang}_{sent_id}"
            result = moderator.moderate(text, user_id)
            decisions[lang] = result["decision"]
            tox_scores[lang] = result.get("toxicity", 0.0)
            threats[lang] = result.get("threat_detected", None)

        unique_decisions = set(decisions.values())
        consistent = len(unique_decisions) == 1
        icon = "" if consistent else ""
        consistency_results.append(consistent)

        # Threat detection summary for threat test cases
        threat_langs = [l for l in LANGUAGES if threats[l] is not None]
        threat_str = ""
        if severity in ("threat", "extreme_threat"):
            threat_str = f"  [threats detected: {len(threat_langs)}/{len(LANGUAGES)}]"

        print(f"\n  {icon} [{severity:>14s}] {sent_id} (expected: {expected}){threat_str}")
        print(f"     Decisions: {unique_decisions}")

        if not consistent:
            for lang in LANGUAGES:
                if decisions[lang] != decisions["en"]:
                    t_flag = f" THREAT:{threats[lang]}" if threats[lang] else ""
                    print(f"     {LANG_NAMES[lang]:12s}: {decisions[lang]:8s} "
                          f"(tox={tox_scores[lang]:.3f}{t_flag}) vs English: {decisions['en']} "
                          f"(tox={tox_scores['en']:.3f})")

        # Show threat detection coverage for threat sentences
        if severity in ("threat", "extreme_threat"):
            missed = [l for l in LANGUAGES if threats[l] is None]
            if missed:
                print(f"     Threat NOT detected in: {', '.join(LANG_NAMES[l] for l in missed)}")
            else:
                print(f"     Threat detected in ALL languages ")

    # Summary
    n_consistent = sum(consistency_results)
    n_total = len(consistency_results)
    print(f"\n{'=' * 80}")
    print(f"SUMMARY: {n_consistent}/{n_total} test cases have consistent decisions across all languages")
    if n_consistent == n_total:
        print(f"   Perfect cross-lingual parity!")
    elif n_consistent >= n_total * 0.7:
        print(f"   Good progress — {n_total - n_consistent} cases still diverge")
    else:
        print(f"   Significant parity gaps remain — {n_total - n_consistent} cases diverge")


if __name__ == "__main__":
    run_parity_test()
    run_threat_detection_test()
    run_production_parity_test()
