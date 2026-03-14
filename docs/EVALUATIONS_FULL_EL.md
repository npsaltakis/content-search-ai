# Συγκεντρωτική Ανάλυση Evaluations

Το αρχείο αυτό συγκεντρώνει όλο το evaluation κομμάτι της πτυχιακής σε ένα σημείο. Περιλαμβάνει τον στόχο των experiments, τη μεθοδολογία, τα completed passes, τα αποτελέσματα, τα θετικά και αρνητικά συμπεράσματα, καθώς και το πώς μπορούν να παρουσιαστούν στην προφορική εξέταση.

---

## 1. Ποιος ήταν ο στόχος των evaluations

Ο βασικός στόχος των evaluations δεν ήταν να αποδείξουμε ότι το σύστημα είναι τέλειο, αλλά να ελέγξουμε αν το threshold που χρησιμοποιείται στο retrieval, δηλαδή η σχέση `mean + 0.3 * std`, είναι λογικό και υποστηρίζεται πειραματικά.

Με πιο πρακτικά λόγια, θέλαμε να απαντήσουμε στα εξής:

- το `0.3` είναι πολύ χαλαρό ή πολύ αυστηρό;
- κόβει χρήσιμα αποτελέσματα;
- αφήνει πολλά αδύναμα ή άσχετα αποτελέσματα;
- στέκει σαν reasonable default baseline;

Άρα τα evaluations αυτά είναι threshold validation experiments και όχι πλήρες benchmark όλου του συστήματος.

---

## 2. Τι ακριβώς συγκρίναμε

Συγκρίναμε τέσσερις εκδοχές threshold:

- `mean`
- `mean + 0.2 * std`
- `mean + 0.3 * std`
- `mean + 0.5 * std`

Η λογική είναι η εξής:

- το `mean` είναι το πιο χαλαρό threshold
- όσο αυξάνεται ο συντελεστής μπροστά από το `std`, το cutoff γίνεται πιο αυστηρό
- όσο πιο αυστηρό το threshold, τόσο λιγότερα αποτελέσματα αφήνει να περάσουν

Το βασικό ερώτημα ήταν αν η αυξημένη αυστηρότητα βελτιώνει ουσιαστικά τα καλύτερα αποτελέσματα ή απλώς μειώνει τον όγκο χωρίς πραγματικό κέρδος ποιότητας.

---

## 3. Τι μετρήσαμε

Σε κάθε evaluation κοιτάξαμε δύο βασικά πράγματα:

### 3.1 Μέγεθος του result set
Πόσα αποτελέσματα περνούν το threshold.

Αυτό δείχνει αν ένα threshold είναι πιο χαλαρό ή πιο αυστηρό.

### 3.2 Συμπεριφορά των κορυφαίων αποτελεσμάτων
Κοιτάξαμε τα top-5 αποτελέσματα για λόγους συγκρισιμότητας.

Σημαντική διευκρίνιση:
- αυτό δεν σημαίνει ότι το σύστημα επιστρέφει πάντα top-5
- σημαίνει ότι στο experiment χρησιμοποιήθηκε `K=5` ως σταθερό evaluation cutoff

Με αυτόν τον τρόπο μπορούσαμε να συγκρίνουμε τα threshold variants πάνω στην ίδια βάση.

---

## 4. Γιατί ξεκινήσαμε με first-pass evaluation

Δεν πήγαμε κατευθείαν σε πλήρες benchmark γιατί θέλαμε ένα πρώτο πείραμα που να είναι:

- ελεγχόμενο
- κατανοητό
- πρακτικό
- thesis-scale
- χωρίς να αλλάξουμε τον retrieval code

Άρα τα experiments έγιναν ως controlled threshold validation passes πάνω στα ήδη indexed δεδομένα.

---

## 5. Completed evaluations μέχρι τώρα

Μέχρι αυτή τη στιγμή έχουν ολοκληρωθεί τρία πραγματικά first-pass evaluations:

- `Text -> PDF`
- `Text -> Image`
- `Image -> Image`

Αυτά είναι τα modalities για τα οποία υπάρχουν πλέον recorded αποτελέσματα και αρχεία υποστήριξης.

Pending ή μερικώς προετοιμασμένα παραμένουν:
- `Text -> Audio`
- `Emotion -> Audio`
- προαιρετικά ένα πιο ξεχωριστό `PDF -> PDF` documented pass

---

## 6. Evaluation για Text -> PDF

Αυτό ήταν το πρώτο και πιο καθαρό evaluation.

### 6.1 Γιατί ξεκινήσαμε από PDFs
Ξεκινήσαμε από PDFs γιατί:

- υπήρχε αρκετό indexed υλικό
- πολλά αρχεία είχαν σαφή ακαδημαϊκή θεματολογία
- μπορούσαμε να χρησιμοποιήσουμε filename/topic proxy για relevance judgment

Δηλαδή αν το query ήταν `decision trees`, ήταν λογικό να αναμένουμε σχετικά αποτελέσματα από PDF που σχετίζονται καθαρά με decision trees.

### 6.2 Query set που χρησιμοποιήθηκε
Χρησιμοποιήθηκαν 10 topic-oriented queries:

- `decision trees`
- `genetic algorithms`
- `neural networks`
- `logistic regression`
- `minmax pruning`
- `ευριστική αναζήτηση`
- `τοπική αναζήτηση`
- `αναζήτηση με αντιπαλότητα`
- `προβλήματα ικανοποίησης περιορισμών`
- `τυφλή αναζήτηση`

### 6.3 Τι έγινε πρακτικά
Για κάθε query:

1. παραγόταν το query embedding
2. συγκρινόταν με όλα τα indexed PDF page embeddings
3. εφαρμόζονταν τα 4 threshold variants
4. καταγραφόταν:
   - πόσες σελίδες περνούσαν το threshold
   - ποια ήταν τα top-5 αποτελέσματα
   - αν τα αποτελέσματα ήταν σχετικά με βάση topic/filename proxy

### 6.4 Αποτελέσματα PDF evaluation
Τα average returned pages per query ήταν:

- `mean`: `746.1`
- `mean + 0.2 * std`: `632.2`
- `mean + 0.3 * std`: `569.6`
- `mean + 0.5 * std`: `440.8`

Στο top-5 παρατηρήθηκαν τα εξής:

- relevant hits in top-5: `8` για όλα τα variants
- queries with at least 1 relevant top-5 result: `4` για όλα τα variants

### 6.5 Ερμηνεία PDF αποτελεσμάτων
Το βασικό συμπέρασμα είναι ότι:

- το `0.3` κόβει αρκετά αποτελέσματα
- αλλά δεν χειροτερεύει το observed top-5 relevance profile
- το αυστηρότερο threshold `0.5` επίσης δεν έδωσε καλύτερο top-5 αποτέλεσμα

Άρα το `mean + 0.3 * std` φαίνεται να είναι ένας λογικός συμβιβασμός:

- πιο αυστηρό από το `mean`
- λιγότερο ακραίο από το `0.5`
- χωρίς εμφανή απώλεια στα κορυφαία αποτελέσματα αυτού του first pass

### 6.6 Θετικά σημεία PDF evaluation
Τα θετικά αποτελέσματα ήταν:

- το `0.3` στέκει πειραματικά ως reasonable default
- το threshold έχει πραγματική επίδραση στο φιλτράρισμα
- το result set μικραίνει αισθητά χωρίς να φαίνεται υποβάθμιση του top-5
- η επιλογή σου παύει να είναι απλώς heuristic χωρίς υποστήριξη

### 6.7 Αρνητικά ή περιορισμοί PDF evaluation
Τα αρνητικά ή οι περιορισμοί ήταν:

- το top-5 relevance profile δεν βελτιώθηκε με πιο αυστηρά thresholds
- άρα το threshold tuning μόνο του δεν λύνει βαθύτερα retrieval προβλήματα
- η αξιολόγηση βασίστηκε σε filename/topic proxy και όχι σε πλήρη page-level annotation
- ήταν first-pass experiment και όχι πλήρης benchmark μελέτη

### 6.8 Συνολική κρίση για PDF evaluation
Το αποτέλεσμα για τα PDFs είναι συνολικά θετικό.

Όχι επειδή αποδείχθηκε ότι το `0.3` είναι βέλτιστο, αλλά επειδή αποδείχθηκε ότι είναι defensible και δεν υπάρχει λόγος άμεσης αλλαγής του με βάση το πρώτο experimental pass.

---

## 7. Evaluation για Text -> Image

Αυτό ήταν το δεύτερο pass και πιο δύσκολο ως προς την ερμηνεία.

### 7.1 Γιατί είναι πιο δύσκολο από το PDF evaluation
Στις εικόνες:

- τα filenames δεν είναι πάντα περιγραφικά
- η σημασιολογική ομοιότητα είναι πιο δύσκολο να κριθεί με αυστηρό τρόπο
- ένα query μπορεί να έχει πολλές οπτικά σχετικές αλλά όχι απολύτως ίδιες απαντήσεις

Άρα εδώ η αξιολόγηση είναι πιο qualitative σε σχέση με τα PDFs.

### 7.2 Γιατί παρ' όλα αυτά αξίζει
Αξίζει γιατί:

- δείχνει ότι το threshold validation δεν έμεινε μόνο σε ένα modality
- δίνει πιο ισχυρή υποστήριξη στην πτυχιακή
- ελέγχει αν η συμπεριφορά του `0.3` είναι συνεπής και στις εικόνες

### 7.3 Query set που χρησιμοποιήθηκε
Χρησιμοποιήθηκαν 5 visually grounded text queries:

- `football player`
- `portrait man`
- `boat`
- `firefighters`
- `band orchestra`

### 7.4 Τι έγινε πρακτικά
Για κάθε image query:

1. παραγόταν το text embedding
2. συγκρινόταν με όλα τα image embeddings
3. εφαρμόζονταν τα 4 threshold variants
4. καταγραφόταν:
   - πόσες εικόνες περνούσαν το threshold
   - ποια ήταν τα top-5 αποτελέσματα
   - αν υπήρχε το αναμενόμενο relevant visual target μέσα στα top results

Επιπλέον δημιουργήθηκαν contact sheets για οπτικό έλεγχο των αποτελεσμάτων.

### 7.5 Αποτελέσματα Text -> Image evaluation
Τα average returned images per query ήταν:

- `mean`: `69.4`
- `mean + 0.2 * std`: `60.2`
- `mean + 0.3 * std`: `53.0`
- `mean + 0.5 * std`: `43.6`

Στο top-5 παρατηρήθηκαν τα εξής:

- relevant hits in top-5: `5` για όλα τα variants
- queries with at least 1 relevant top-5 result: `5` για όλα τα variants

### 7.6 Ερμηνεία Text -> Image αποτελεσμάτων
Το βασικό συμπέρασμα είναι ότι και στις εικόνες:

- το `0.3` μείωσε το πλήθος των αποδεκτών αποτελεσμάτων
- χωρίς να χαλάσει το observed top-5 relevance proxy στο συγκεκριμένο sample
- το αυστηρότερο threshold `0.5` δεν έδωσε καλύτερο top-5 αποτέλεσμα

Άρα και εδώ το `mean + 0.3 * std` φαίνεται reasonable middle-ground default.

### 7.7 Θετικά σημεία Text -> Image evaluation
Τα θετικά αποτελέσματα ήταν:

- το `0.3` πάλι στάθηκε καλά σαν threshold
- η συμπεριφορά του threshold ήταν συνεπής με το PDF pass
- μειώθηκε ο αριθμός των αποτελεσμάτων χωρίς απώλεια στον top-5 proxy στόχο
- η πτυχιακή πλέον έχει threshold validation σε περισσότερα από ένα modalities

### 7.8 Αρνητικά ή περιορισμοί Text -> Image evaluation
Το image evaluation έδειξε και πιο καθαρά ένα όριο:

- σε broad queries όπως `portrait man` ή `firefighters` εμφανίστηκε semantic drift
- δηλαδή κάποια top results ήταν σχετικά μόνο γενικά και όχι απολύτως ακριβή

Αυτό σημαίνει ότι:

- το threshold μόνο του δεν λύνει semantic ambiguity
- η ποιότητα εξαρτάται και από το ίδιο το embedding/model behavior
- η image evaluation είναι πιο noisy από το PDF evaluation
- η relevance κρίση χρειάζεται visual inspection και όχι μόνο filename proxy

### 7.9 Συνολική κρίση για Text -> Image evaluation
Το αποτέλεσμα για τις εικόνες είναι θετικό αλλά πιο προσεκτικό.

Θετικό γιατί:
- το `0.3` επιβεβαιώθηκε ως practical default

Πιο προσεκτικό γιατί:
- το image retrieval έχει εντονότερες σημασιολογικές ασάφειες
- και αυτές δεν λύνονται μόνο με threshold tuning

---

## 8. Evaluation για Image -> Image

Αυτό ήταν το τρίτο pass και ήρθε να συμπληρώσει το image κομμάτι της αξιολόγησης.

### 8.1 Γιατί είχε αξία να γίνει
Το `Image -> Image` είναι βασικός τρόπος αναζήτησης του συστήματος, άρα είχε αξία να ελεγχθεί και αυτό πειραματικά και όχι μόνο το `Text -> Image`.

Επιπλέον:
- υπήρχαν ήδη indexed εικόνες με αρκετά διακριτά οπτικά μοτίβα
- μπορούσε να στηθεί ένα μικρό manual pass χωρίς να αλλάξει καθόλου το UI ή ο retrieval code
- έτσι ενισχύθηκε η πολυτροπική αξιολόγηση της πτυχιακής

### 8.2 Πώς στήθηκε το πείραμα
Χρησιμοποιήθηκαν 5 representative query images, τα οποία αντιγράφηκαν έξω από το indexed archive ώστε το query να μην περάσει απλώς σαν ίδιο path που ήδη υπάρχει στη βάση.

Τα query images βασίστηκαν σε αρχεία όπως:
- `Lionel_Messi.jpg`
- `407869068_10161463377388069_270481176185833064_n2.jpg`
- `89332423.jpg`
- `89404014.jpg`
- `89407459.jpg`

Για κάθε query ορίστηκε ένα μικρό relevance proxy, δηλαδή ένα σύνολο από clearly relevant archive images, όπως exact matches, near-duplicates ή πολύ κοντινές οπτικές παραλλαγές.

### 8.3 Τι έγινε πρακτικά
Για κάθε query image:

1. υπολογίστηκε image embedding με το ίδιο retrieval pipeline
2. συγκρίθηκε με όλα τα stored image embeddings
3. εφαρμόστηκαν τα 4 threshold variants
4. καταγράφηκαν:
   - πόσα αποτελέσματα περνούσαν το threshold
   - ποια ήταν τα top-5 αποτελέσματα
   - αν τα top-5 περιείχαν τα αναμενόμενα relevant visual targets

Επιπλέον δημιουργήθηκαν contact sheets για οπτικό έλεγχο των top αποτελεσμάτων.

### 8.4 Αποτελέσματα Image -> Image evaluation
Τα average returned images per query ήταν:

- `mean`: `73.8`
- `mean + 0.2 * std`: `56.6`
- `mean + 0.3 * std`: `49.6`
- `mean + 0.5 * std`: `37.0`

Στο top-5 παρατηρήθηκαν τα εξής:

- relevant hits in top-5: `8` για όλα τα variants
- queries with at least 1 relevant top-5 result: `5` για όλα τα variants

### 8.5 Ερμηνεία Image -> Image αποτελεσμάτων
Το βασικό συμπέρασμα είναι ότι και στο `Image -> Image`:

- το `0.3` μείωσε ξανά αισθητά το μέγεθος του result set
- χωρίς να χειροτερεύσει το observed top-5 relevance proxy στο sample
- το αυστηρότερο threshold `0.5` δεν έφερε καλύτερο top-5 αποτέλεσμα

Άρα και εδώ το `mean + 0.3 * std` λειτούργησε ως reasonable middle-ground default.

### 8.6 Θετικά σημεία Image -> Image evaluation
Τα θετικά αποτελέσματα ήταν:

- το `0.3` επιβεβαιώθηκε για τρίτη φορά ως πρακτικά λογικό threshold
- η συμπεριφορά του threshold ήταν συνεπής και σε αυτό το retrieval mode
- το result set μειώθηκε αισθητά χωρίς απώλεια στο top-5 proxy αποτέλεσμα
- η πτυχιακή έχει πλέον threshold validation σε τρία retrieval settings

### 8.7 Αρνητικά ή περιορισμοί Image -> Image evaluation
Οι βασικοί περιορισμοί ήταν:

- χρησιμοποιήθηκαν μόνο 5 query images
- το relevance proxy ήταν μικρό και συντηρητικό
- κάποια visually plausible matches μπορεί να ήταν λογικά, αλλά να μην ανήκαν στο προκαθορισμένο relevant set
- όπως και στα άλλα passes, το threshold tuning από μόνο του δεν λύνει πλήρως την ασάφεια της visual similarity

### 8.8 Συνολική κρίση για Image -> Image evaluation
Το αποτέλεσμα για το `Image -> Image` είναι θετικό.

Το πείραμα στήριξε ξανά την ίδια βασική ιδέα:
- το `0.3` λειτουργεί καλά ως practical filtering threshold
- αλλά δεν πρέπει να παρουσιάζεται σαν μαγική παράμετρος που λύνει από μόνη της όλα τα retrieval ζητήματα

---

## 9. Συνολική αποτίμηση όλων των evaluations

Αν δούμε μαζί το PDF, το Text -> Image και το Image -> Image pass, τα συνολικά συμπεράσματα είναι τα εξής:

### 9.1 Τι έδειξαν θετικά
Τα evaluations έδειξαν ότι:

1. το `mean + 0.3 * std` είναι reasonable baseline
2. η αύξηση του threshold μειώνει ουσιαστικά το πλήθος των αποδεκτών αποτελεσμάτων
3. το `0.3` δεν χειροτέρεψε το observed top-5 profile στα three first-pass experiments
4. δεν υπάρχει πειραματική ένδειξη ότι κάποιο αυστηρότερο threshold είναι άμεσα καλύτερο
5. η τωρινή επιλογή threshold μπορεί πλέον να υποστηριχθεί πιο σοβαρά στην πτυχιακή

### 9.2 Τι έδειξαν αρνητικά ή περιοριστικά
Τα evaluations έδειξαν επίσης ότι:

1. το threshold tuning μόνο του δεν βελτιώνει από μόνο του τη βαθύτερη semantic quality
2. αν υπάρχει semantic mismatch, δεν λύνεται απλώς με αυστηρότερο cutoff
3. η ποιότητα retrieval εξαρτάται και από τα embeddings, το dataset και τη φύση του query
4. τα experiments είναι first-pass και όχι πλήρες multimodal benchmark

### 9.3 Άρα είναι τελικά θετικά ή αρνητικά τα αποτελέσματα;
Η σωστή απάντηση είναι ότι είναι συνολικά περισσότερο θετικά παρά αρνητικά.

Θετικά γιατί:
- επιβεβαιώνουν την υπάρχουσα επιλογή threshold
- στηρίζουν τον σχεδιασμό του retrieval pipeline

Αρνητικά μόνο με την έννοια ότι:
- έδειξαν και τα όρια του threshold tuning
- άρα άνοιξαν τον δρόμο για μελλοντικές βελτιώσεις πέρα από το cutoff

---

## 10. Πώς να το πεις στην προφορική

Μια καλή σύντομη απάντηση είναι η εξής:

> Έγινε first-pass experimental validation του adaptive threshold σε τρία retrieval settings, πρώτα στο Text-to-PDF, μετά στο Text-to-Image και στη συνέχεια στο Image-to-Image. Συγκρίθηκαν τέσσερις εκδοχές threshold, από το `mean` μέχρι το `mean + 0.5 * std`. Και στα τρία πειράματα, η αύξηση του threshold μείωσε σημαντικά το πλήθος των αποδεκτών αποτελεσμάτων, αλλά δεν βελτίωσε το observed top-5 relevance profile. Αυτό σημαίνει ότι το `mean + 0.3 * std` λειτουργεί ως λογικό πρακτικό default: φιλτράρει πιο αδύναμα αποτελέσματα χωρίς να υποβαθμίζει τα κορυφαία matches στο πρώτο experimental pass.

---

## 11. Τι να πεις αν σε ρωτήσουν αν τα experiments ήταν επιτυχία

Μια πολύ καλή απάντηση είναι:

> Τα experiments ήταν επιτυχία ως validation της τωρινής επιλογής threshold, όχι ως απόδειξη ότι λύθηκε συνολικά το retrieval optimization πρόβλημα.

Αυτό είναι ώριμη και σωστή απάντηση.

---

## 12. Τι να πεις αν σε ρωτήσουν ποιο είναι το βασικό δίδαγμα

Το βασικό δίδαγμα των evaluations είναι το εξής:

> Το `mean + 0.3 * std` είναι ένα defensible adaptive threshold baseline για το τωρινό project scale, αλλά η βελτίωση του retrieval δεν εξαρτάται μόνο από το threshold. Για μεγαλύτερη βελτίωση χρειάζονται και καλύτερα embeddings, richer evaluation sets και πιο εκτεταμένη multimodal αξιολόγηση.

---

## 13. Τελικό συμπέρασμα

Τα evaluations που έγιναν μέχρι στιγμής:

- δεν απέδειξαν ότι το `0.3` είναι το παγκόσμια βέλτιστο threshold
- απέδειξαν όμως ότι είναι μια λογική, αμυντικά σωστή και πειραματικά υποστηριζόμενη default επιλογή
- έδειξαν ότι stricter thresholds μικραίνουν το result set
- έδειξαν ότι αυτό δεν μεταφράζεται αυτόματα σε καλύτερα top results
- έδωσαν στην πτυχιακή ένα πιο σοβαρό experimental υπόβαθρο

Με μία φράση:

> Τα evaluations ήταν χρήσιμα, συνολικά θετικά, και αρκετά ώστε να υποστηρίξουν την τωρινή threshold επιλογή στην προφορική εξέταση.

---

## 14. Σχετικά αρχεία

Τα βασικά αρχεία που υποστηρίζουν το evaluation κομμάτι είναι:

- `docs/EVALUATION_PLAN.md`
- `docs/EVALUATION_RESULTS.md`
- `docs/ORAL_EVALUATION_NOTES_EL.md`
- `evaluation/EVALUATION_STATUS.md`
- `evaluation/pdf_threshold_summary.csv`
- `evaluation/pdf_threshold_top5.csv`
- `evaluation/image_threshold_summary.csv`
- `evaluation/image_threshold_top5.csv`
- `evaluation/image_to_image_threshold_summary.csv`
- `evaluation/image_to_image_threshold_top5.csv`
- `evaluation/image_eval_sheets/`
- `evaluation/image_to_image_eval_sheets/`
