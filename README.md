 # Vítejte v aplikaci Vocab2Anki

 ## O aplikaci

Aplikace slouží k vytvoření souboru s česko-anglickými slovíčky ve formátu umožnující importaci do aplikace ANKI.
Jako vstupní soubor pro aplikaci je možné použít pdf, nebo epub obsahující anglický text.

Aplikace nejdříve načte vstupní soubor a text uvnitř rozdělí na jednotlivá slova, která převede na základní tvar, 
seřadí dle četnosti výskytu, odstraní vlastní jména a obecně neanglická slova.
Slova dále rozdělí dle jazykové úrovně na "A1", "A2", "B1", "B2", "C1", "C2", "C2+".
Dle uživatelova nastavení vybere počet slov z určitých jazykových úrovní a k těmto slovům přidá překlad + výslovnost a vytvoří soubor s názvem "ANKI_soubor.csv"
V případě, že uživatel požaduje menší počet slov, než kolik se jich ve vstupním dokumentu nachází, aplikace upřednostní slova,
která mají ve vstupním dokumentu vyšší četnost.

Aplikaci je možné využít také k vytvoření souboru se všemi slovíčky potřebnými pro jednotlivé úrovně angličtiny. Tyto soubory jsem již připravil – můžete si je stáhnout z tohoto repozitáře a importovat do aplikace Anki.



