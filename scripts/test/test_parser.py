from unittest import TestCase
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))
from scripts.parser.parser import CDRParser
from scripts.models.models import Document


class TestCDRParser(TestCase):

    def test_parser(self):
        parser = CDRParser()
        title = 'Naloxone reverses the antihypertensive effect of clonidine.'
        abtract = 'In unanesthetized, spontaneously hypertensive rats the decrease in blood pressure and heart rate produced by intravenous clonidine, 5 to 20 micrograms/kg, was inhibited or reversed by nalozone, 0.2 to 2 mg/kg. The hypotensive effect of 100 mg/kg alpha-methyldopa was also partially reversed by naloxone. Naloxone alone did not affect either blood pressure or heart rate. In brain membranes from spontaneously hypertensive rats clonidine, 10(-8) to 10(-5) M, did not influence stereoselective binding of [3H]-naloxone (8 nM), and naloxone, 10(-8) to 10(-4) M, did not influence clonidine-suppressible binding of [3H]-dihydroergocryptine (1 nM). These findings indicate that in spontaneously hypertensive rats the effects of central alpha-adrenoceptor stimulation involve activation of opiate receptors. As naloxone and clonidine do not appear to interact with the same receptor site, the observed functional antagonism suggests the release of an endogenous opiate by clonidine or alpha-methyldopa and the possible role of the opiate in the central control of sympathetic tone.'
        document = Document(title=title,
                            abstract=abtract,
                            list_cid=None,
                            list_entity=None)
        parser.parse(doc=document)
