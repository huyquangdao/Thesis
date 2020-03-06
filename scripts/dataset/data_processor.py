import sys
from pathlib import Path
import copy
sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))
import itertools
from scripts.models import models
from collections import defaultdict
import scripts.models.constants as constants
import scripts.dataset.pre_process as pre_process
from scripts.dataset.feature_engineering.deptree.parsers import SpacyParser
from scripts.dataset.feature_engineering.deptree.sdp import Finder
from scripts.dataset.pre_process import opt as pre_opt
from scripts.dataset.data_readers import CDRReader
from scripts.dataset.data_managers import CDRDataManager
from scripts.dataset.feature_engineering.deptree.models import DepTree
import pickle
from sklearn.utils import shuffle


class BaseProcessor(object):

    def __init__(self):
        pass


class CDRProcessor(BaseProcessor):

    def __init__(self, file_path):
        super(CDRProcessor, self).__init__()
        self.pre_config = {
            pre_opt.SEGMENTER_KEY: pre_opt.SpacySegmenter(),
            pre_opt.TOKENIZER_KEY: pre_opt.SpacyTokenizer()
        }
        self.spd_finder = Finder()
        self.parser = SpacyParser()
        self.reader = CDRReader(file_name=file_path)
        self.data_manager = CDRDataManager()

    def __check_in_sentence(self, sent, entity):

        return sent.doc_offset[0] <= entity.tokens[0].doc_offset[0] < entity.tokens[-1].doc_offset[1] < sent.doc_offset[
            1]

    def __get_entities(self, entities, type):
        list_entities = []
        for entity in entities:
            if entity.type == type:
                list_entities.append(entity)
        return list_entities

    def __get_pairs_entities_same_sentence(self, curr_entities):

        chem_list = []
        dis_list = []

        for entity in curr_entities:
            if entity.type == constants.ENTITY_TYPE_CHEMICAL:
                chem_list.append(entity)
            elif entity.type == constants.ENTITY_TYPE_DISEASE:
                dis_list.append(entity)

        return list(itertools.product(chem_list, dis_list))

    def _get_all_pairs_entities_shortert_than_5(self,abstract, entities , curr_index):
        list_pairs_entities = []
        for i in range(len(abstract) - 1):
            curr_sent_entities = [entity for entity in entities if self.__check_in_sentence(abstract[i], entity)]
            if len(curr_sent_entities) == 0:
                continue
            else:
                all_pairs_entities_same_sent = self.__get_pairs_entities_same_sentence(curr_sent_entities)

                sent = [[curr_index + i, curr_index + i]] * len(all_pairs_entities_same_sent)

                dic_info = {'sent':sent,'pairs':all_pairs_entities_same_sent}

                if len(all_pairs_entities_same_sent) > 0:
                    list_pairs_entities.append(dic_info)

                curr_chemical_entities = self.__get_entities(curr_sent_entities,
                                                                 type=constants.ENTITY_TYPE_CHEMICAL)
                curr_disease_entities = self.__get_entities(curr_sent_entities,
                                                                type=constants.ENTITY_TYPE_DISEASE)
                for j in range(i + 1, len(abstract)):
                    next_sent_entities = [entity for entity in entities if
                                          self.__check_in_sentence(abstract[j], entity)]
                    next_disease_entities = self.__get_entities(next_sent_entities,
                                                                type=constants.ENTITY_TYPE_DISEASE)
                    next_chemical_entites = self.__get_entities(next_sent_entities,
                                                                type=constants.ENTITY_TYPE_CHEMICAL)

                    all_pairs_different_sent = []
                    sent = []
                    if len(next_disease_entities) > 0:
                        temp = list(
                            itertools.product(curr_chemical_entities, next_disease_entities))
                        sent.extend([[curr_index + i, curr_index + j]]*len(temp))
                        all_pairs_different_sent.extend(temp)
                    # if len(next_chemical_entites) > 0:
                    #     temp = list(itertools.product(next_chemical_entites, curr_disease_entities))
                    #     sent.extend([[curr_index + j, curr_index + i]] * len(temp))
                    #     all_pairs_different_sent.extend(temp)
                    if len(all_pairs_different_sent) > 0:
                        dic_info = {'sent':sent,'pairs':all_pairs_different_sent}
                        list_pairs_entities.append(dic_info)
                    if j == len(abstract) - 1:
                        all_same_pairs_last = self.__get_pairs_entities_same_sentence(next_sent_entities)
                        sent = [[ curr_index + j, curr_index + j]] * len(all_same_pairs_last)
                        dic_info = {'sent': sent, 'pairs': all_same_pairs_last}
                        if len(all_same_pairs_last) > 0:
                            list_pairs_entities.append(dic_info)
        # print(list_pairs_entities)
        return list_pairs_entities

    def process_sentences(self, sentences, entities):
        list_pairs_entities = []
        abstract = sentences[1:]
        # print(len(abstract))
        if len(abstract) <= 4:
            list_pairs_entities.extend(self._get_all_pairs_entities_shortert_than_5(abstract,entities, curr_index= 0))

        else:
            for i in range(len(abstract) - 3):
                curr_sent_entities = [entity for entity in entities if self.__check_in_sentence(abstract[i], entity)]
                if i == len(abstract) - 4:
                    list_pairs_entities.extend(self._get_all_pairs_entities_shortert_than_5(abstract[i:], entities, curr_index = i))
                    break
                if len(curr_sent_entities) == 0:
                    continue
                else:
                    all_pairs_entities_same_sent = self.__get_pairs_entities_same_sentence(curr_sent_entities)
                    if len(all_pairs_entities_same_sent) > 0:

                        sent = [[i, i]] * len(all_pairs_entities_same_sent)

                        dic_info = {'sent': sent, 'pairs': all_pairs_entities_same_sent}

                        list_pairs_entities.append(dic_info)

                    curr_chemical_entities = self.__get_entities(curr_sent_entities,
                                                                 type=constants.ENTITY_TYPE_CHEMICAL)
                    curr_disease_entities = self.__get_entities(curr_sent_entities,
                                                                type=constants.ENTITY_TYPE_DISEASE)
                    for j in range(i + 1, i + 4):

                        next_sent_entities = [entity for entity in entities if
                                              self.__check_in_sentence(abstract[j], entity)]
                        next_disease_entities = self.__get_entities(next_sent_entities,
                                                                    type=constants.ENTITY_TYPE_DISEASE)
                        next_chemical_entites = self.__get_entities(next_sent_entities,
                                                                    type=constants.ENTITY_TYPE_CHEMICAL)

                        all_pairs_different_sent = []

                        sent = []

                        if len(next_disease_entities) > 0:
                            temp = list(
                                itertools.product(curr_chemical_entities, next_disease_entities))
                            sent.extend([[i,j]] * len(temp))
                            all_pairs_different_sent.extend(temp)
                        # if len(next_chemical_entites) > 0:
                        #     temp = list(
                        #         itertools.product(next_chemical_entites, curr_disease_entities))
                        #     sent.extend([[j, i]] * len(temp))
                        #     all_pairs_different_sent.extend(temp)
                        if len(all_pairs_different_sent) > 0:
                            # sent = [[i, j]] * len(all_pairs_different_sent)
                            dic_info = {'sent': sent, 'pairs': all_pairs_different_sent}
                            list_pairs_entities.append(dic_info)

        return list_pairs_entities

    def __process_one(self, doc):
        a = list()
        for sent in doc.sentences:
            deptree = self.parser.parse(sent)
            a.append(deptree)
        return a

    def generate_data(self):

        raw_documents = self.reader.read()
        raw_entities = self.reader.read_entity()
        raw_relations = self.reader.read_relation()

        # title_docs, abstract_docs = self.data_manager.parse_documents(raw_documents)
        #
        # # Pre-process
        # title_doc_objs = pre_process.process(title_docs, self.pre_config, constants.SENTENCE_TYPE_TITLE)
        # abs_doc_objs = pre_process.process(abstract_docs, self.pre_config, constants.SENTENCE_TYPE_ABSTRACT)
        # documents = self.data_manager.merge_documents(title_doc_objs, abs_doc_objs)

        with open('../../data/cdr_data/documents.pkl', 'rb') as f:
            documents = pickle.load(f)

        # Generate data
        dict_nern = defaultdict(list)
        data_tree = defaultdict()
        for doc in documents:
            raw_entity = raw_entities[doc.id]

            # print(raw_entity)

            for r_en in raw_entity:
                entity_obj = models.BioEntity(tokens=[], ids={})
                entity_obj.content = r_en[3]
                entity_obj.type = constants.ENTITY_TYPE_CHEMICAL if r_en[4] == "Chemical" \
                    else constants.ENTITY_TYPE_DISEASE
                entity_obj.ids[constants.MESH_KEY] = r_en[5]

                for s in doc.sentences:
                    if s.doc_offset[0] <= int(r_en[1]) < s.doc_offset[1]:
                        for tok in s.tokens:
                            if (int(r_en[1]) <= tok.doc_offset[0] < int(r_en[2])
                                    or int(r_en[1]) < tok.doc_offset[1] <= int(r_en[2])
                                    or tok.doc_offset[0] <= int(r_en[1]) < int(r_en[2]) <= tok.doc_offset[1]):
                                entity_obj.tokens.append(tok)
                if len(entity_obj.tokens) == 0:
                    print(doc.id, r_en)

                dict_nern[doc.id].append(entity_obj)

            # dep_tree = self.__process_one(doc)
            # data_tree[doc.id] = dep_tree

        with open('../../data/cdr_data/data_tree.pkl','rb') as f:
            data_tree = pickle.load(f)

        print(len(documents))
        with open('../../data/processed/sdp_train' + ".txt", "w") as f:
            for i,doc in enumerate(documents):
                print(i)
                sdp_data = defaultdict(dict)
                deep_tree_doc = data_tree[doc.id]
                deep_tree_doc = deep_tree_doc[1:]
                relation = raw_relations[doc.id]
                list_dic_info = self.process_sentences(doc.sentences, dict_nern[doc.id])
                if len(list_dic_info) ==0:
                    continue
                else:
                    for dic_info in list_dic_info:
                        sent_pairs = dic_info['sent']
                        en_pairs = dic_info['pairs']
                        for sent_pair, (chem_en,dis_en) in zip(sent_pairs,en_pairs):
                            chem_idx = sent_pair[0]
                            dis_idx = sent_pair[1]
                            chem_token = chem_en.tokens[-1]
                            dis_token = dis_en.tokens[-1]
                            if chem_idx == dis_idx:

                                sent = doc.sentences[chem_idx + 1]
                                sent_offset2idx = {}
                                for idx, token in enumerate(sent.tokens):
                                    sent_offset2idx[token.sent_offset] = idx
                                dep_tree = deep_tree_doc[chem_idx]
                                r_path = self.spd_finder.find_sdp(dep_tree,chem_token,dis_token)
                                new_r_path = copy.deepcopy(r_path)

                                for i, x in enumerate(new_r_path):
                                    if i % 2 == 0:
                                        x.content += "_" + str(sent_offset2idx[x.sent_offset])
                                path = self.spd_finder.parse_directed_sdp(new_r_path)

                                sent_path = '|'.join([token.content for token in sent.tokens])
                                if path:
                                    chem_ids = chem_en.ids[constants.MESH_KEY].split('|')
                                    dis_ids = dis_en.ids[constants.MESH_KEY].split('|')
                                    rel = 'CID'
                                    for chem_id, dis_id in itertools.product(chem_ids, dis_ids):
                                        if (doc.id, 'CID', chem_id, dis_id) not in relation:
                                            rel = 'NONE'
                                            break
                                    for chem_id, dis_id in itertools.product(chem_ids, dis_ids):
                                        key = '{}_{}'.format(chem_id, dis_id)
                                        if rel not in sdp_data[key]:
                                            sdp_data[key][rel] = []
                                        sdp_data[key][rel].append([path, sent_path])

                            else:
                                chem_edges = deep_tree_doc[chem_idx]
                                dis_edges = deep_tree_doc[dis_idx]
                                chem_tree = DepTree(edges=chem_edges)
                                dis_tree = DepTree(edges=dis_edges)

                                chem_root = chem_tree.get_root()
                                dis_root = dis_tree.get_root()

                                chem_sdp = self.spd_finder.find_sdp(chem_edges,chem_token,chem_root)
                                dis_sdp = self.spd_finder.find_sdp(dis_edges,dis_root,dis_token)

                                chem_sent = doc.sentences[chem_idx + 1]
                                dis_sent = doc.sentences[dis_idx + 1]
                                chemsent_offset2idx = {}
                                for idx, token in enumerate(chem_sent.tokens):
                                    chemsent_offset2idx[token.sent_offset] = idx
                                dissent_offset2idx = {}
                                for idx, token in enumerate(dis_sent.tokens):
                                    dissent_offset2idx[token.sent_offset] = idx
                                new_chem_path = copy.deepcopy(chem_sdp)
                                new_dis_path = copy.deepcopy(dis_sdp)
                                for i, x in enumerate(new_chem_path):
                                    if i % 2 == 0:
                                        x.content += "_" + str(chemsent_offset2idx[x.sent_offset])
                                for i, x in enumerate(new_dis_path):
                                    if i % 2 == 0:
                                        x.content += "_" + str(dissent_offset2idx[x.sent_offset])

                                chem_path = self.spd_finder.parse_directed_sdp(new_chem_path)
                                dis_path = self.spd_finder.parse_directed_sdp(new_dis_path)

                                if chem_path and dis_path:

                                    if abs(chem_idx - dis_idx) > 1:
                                        between_root = []
                                        mi = min(chem_idx, dis_idx)
                                        ma = max(chem_idx, dis_idx)
                                        for i in range(mi + 1, ma):
                                            edges = deep_tree_doc[i]
                                            try:

                                                tree = DepTree(edges=edges)
                                                root = tree.get_root()
                                                sent = doc.sentences[i + 1]
                                                root_edged = None
                                                for idx, token in enumerate(sent.tokens):
                                                    if token.doc_offset == root.doc_offset:
                                                        root_edged = root.content + '_' + str(idx) + '/' + root.metadata[
                                                            'pos_tag'] + ' (r_none)'
                                                        break
                                                between_root.append(root_edged)
                                            except:
                                                print('in here')
                                                between_root = []

                                        between_root = ' '.join(between_root)
                                        sdp_path = chem_path + ' (r_none) ' + between_root + ' ' + dis_path
                                    else:
                                        sdp_path = chem_path + ' (r_none) ' + dis_path
                                    chem_ids = chem_en.ids[constants.MESH_KEY].split('|')
                                    dis_ids = dis_en.ids[constants.MESH_KEY].split('|')
                                    rel = 'CID'
                                    for chem_id, dis_id in itertools.product(chem_ids, dis_ids):
                                        if (doc.id, 'CID', chem_id, dis_id) not in relation:
                                            rel = 'NONE'
                                            break

                                    for chem_id, dis_id in itertools.product(chem_ids, dis_ids):
                                        key = '{}_{}'.format(chem_id, dis_id)

                                        if rel not in sdp_data[key]:
                                            sdp_data[key][rel] = []

                                        sdp_data[key][rel].append([sdp_path, '_'])
                # for key,value in sdp_data.items():
                #     print(key,value)

                for pair_key in sdp_data:
                    if 'CID' in sdp_data[pair_key]:
                        for k in range(len(sdp_data[pair_key]['CID'])):
                            sdp, sent_path = sdp_data[pair_key]['CID'][k]
                            f.write('{} {} {}\n'.format(pair_key, 'CID', sdp))

                    if 'NONE' in sdp_data[pair_key]:
                        for k in range(len(sdp_data[pair_key]['NONE'])):
                            sdp, sent_path = sdp_data[pair_key]['NONE'][k]
                            f.write('{} {} {}\n'.format(pair_key, 'NONE', sdp))




