'''
Author: Kenyi Despean
Rev: 2.1
Last Modified time: 2020-04-24
Model Load seperated from model initailization
'''

import re
import usaddress
import uuid
import nltk
import json
import allennlp
import pprint
import traceback
import sys
import pyap
from address_parser import idea_address_parser

from nameparser import HumanName
from ip2geotools.databases.noncommercial import DbIpCity

from nltk.tokenize import sent_tokenize
from allennlp.predictors.predictor import Predictor

import warnings
warnings.filterwarnings("ignore")

nltk.data.path.append('nltk_data/')  # set the path
#nltk.download('punkt')

class Models:
    def __init__(self):

        # Load NER Model
        self.ner_predictor = Predictor.from_path(
              #"kenyi-model-2020.03.31.tar.gz"
             "model-2020.08.24.tar.gz"
             )

        # Load Coreference Model
        self.coref_predictor = Predictor.from_path(
             #"coref-model-2018.02.05.tar.gz"
            "coref-spanbert-large-2020.02.27.tar.gz"
            )

        self.srl_predictor = Predictor.from_path(
             #"srl-model-2018.05.25.tar.gz"
            "bert-base-srl-2020.03.24.tar.gz"
            )

        self.ethnicity_list = ["Asian", "Black", "Caucasian", "Eur-Asian", "First Nation", "Hispanic", "Indian not FN",
                               "Inuit", "Metis", "Middle Eastern", "Mongoloid", "Other Non-White"]

        self.address_parse_dict = {
            "AddressNumberPrefix": "streetNumber", # a modifier before an address number, e.g. 'Mile', '#'
            "AddressNumber": "streetNumber", # address number
            "AddressNumberSuffix": "streetNumber", # a modifier after an address number, e.g 'B', '1/2'
            "BuildingName": "buildingName", # the name of a building, e.g. 'Atlanta Financial Center'
            "CornerOf": "notes", # words indicating that an address is a corner, e.g. 'Junction', 'corner of'
            "IntersectionSeparator": "", # a conjunction connecting parts of an intersection, e.g. 'and', '&'
            "LandmarkName": "notes", # the name of a landmark, e.g. 'Wrigley Field', 'Union Station'
            "NotAddress": "notes", # a non-address component that doesn't refer to a recipient
            "OccupancyType": "apartmentNo", # a type of occupancy within a building, e.g. 'Suite', 'Apt', 'Floor'
            "OccupancyIdentifier": "apartmentNo", # the identifier of an occupancy, often a number or letter
            "PlaceName": "city", # city
            "CountryName": "country", #country
            "Recipient": "notes", # a non-address recipient, e.g. the name of a person/organization
            "StateName": "provinceState", # state
            "StreetNamePreDirectional": "streetName", # a direction before a street name, e.g. 'North', 'S'
            "StreetNamePreModifier": "streetName", # a modifier before a street name that is not a direction, e.g. 'Old'
            "StreetNamePreType": "streetName", # a street type that comes before a street name, e.g. 'Route', 'Ave'
            "StreetName": "streetName", # street name, excluding type & direction
            "StreetNamePostModifier": "streetName", # a modifier after a street name, e.g. 'Ext'
            "StreetNamePostDirectional": "streetDirection", # a direction after a street name, e.g. 'North', 'S'
            "StreetNamePostType": "streetType", # a street type that comes after a street name, e.g. 'Avenue', 'Rd'
            "SubaddressIdentifier": "notes", # the name/identifier of a subaddress component
            "SubaddressType": "notes", # a level of detail in an address that is not an occupancy within a building, e.g. 'Building', 'Tower'
            "USPSBoxGroupID": "poBox", # the identifier of a USPS box group, usually a number
            "USPSBoxGroupType": "poBox", # a name for a group of USPS boxes, e.g. 'RR'
            "USPSBoxID": "poBox", # the identifier of a USPS box, usually a number
            "USPSBoxType": "", # a USPS box, e.g. 'P.O. Box'
            "ZipCode": "postCode", # zip code
        }
        
        self.address_schema_map_direction = {
            "WEST":"W",
            "EAST":"E",
            "SOUTH":"S",
            "NORTH":"N"
        }
        
        self.addressStreetTypeOptions={
            "St":"Street"
        }
        
        self.reason = ["Alias", "Associate", "Associated Business", "Beneficiary", "Brother", "Business",
                       "Business Owner", "Cash Flow", "Complainant", "Complainant Address", "Complainant Email",
                       "Complainant Phone", "Correspondent Bank", "Cousin", "Current Address", "Current Email Address",
                       "Current License Plate", "Current Phone Number", "Destination Bank", "Displayed On",
                       "Driver", "Editing Investigator", "Frequents No Fixed Address", "Incident Location",
                       "Involved in Ticket", "Issued To", "Issuer", "Observed", "Origin Bank", "Other",
                       "Owner", "Payee", "Payee Bank", "Payor", "Payor Bank", "Permit Element", "Person Owner",
                       "Physical location", "Registered Owner", "Residence", "Seasonal Residence", "Sister",
                       "Temporary residence", "Theft Of", "Ticketed Entity", "Ticketing Officer Note", "Uses"]

class Extract:
    # Initial the extractor for each parse
    def __init__(self, models) :
 
        self.models = models
        self.person = []
        self.ethnicity = []
        self.Named_entity = []

        self.person_check = []
        self.entity_check = []

        self.group_list = []
        self.antecedent_list = []
        self.entity_list = []

        self.all_entity = {}

        self.data = []

        self.relationship = []

        # Testing run
        self.common_list = []


    # Map one value to another using a dictionary, returning the original value if not mapped
    def map_term(self, obj, attr, dict):
        if attr in obj:
            new_attr = dict.get(obj[attr].upper())
            if new_attr != None: 
                obj[attr] = new_attr


    # Define Extraction using Regex
    def extract_email(self, string):
        r = re.compile(r'[\w\.-]+@[\w\.-]+')
        return r.findall(string)

    def extract_domain(self, string):
        regex = r'[\w\.-]+@[\w\.-]+'
        result = re.sub(regex, '', string, 0)
        r = re.compile(
            r'(?:[a-zA-Z0-9_-](?:[a-zA-Z0-9\-_-]{,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,6}')
        return r.findall(result)

    def extract_weight(self, string):
        r = re.compile(r'\d+(?:lbs|kgs)')
        return r.findall(string)

    def extract_height(self, string):
        r = re.compile(r'[0-9]+\'[0-9]{2}')
        return r.findall(string)

    def extract_ip(self, string):
        r = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
        r2 = re.compile(r'(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))')
        n = r2.findall(string)
        ip_list =[]
        for ip in n:
            ip_list.append(ip[0])
        return r.findall(string) + ip_list

    def extract_card_number(self, string):
        r = re.compile(r'(?:[0-9]{4}-){3}[0-9]{4}|[0-9]{16}')
        card = r.findall(string)
        return card

    def extract_phone_number(self, string):
        r = re.compile(r"\b((1\W*)?([2-9][0-8][0-9])\W*(([2-9][0-9]{2})\W*([0-9]{4})))\b")
        r2 = re.compile(r'\b((5[0-9][0-9]?\s?)?(\([0-9]{2}\)|[0-9]{2}[\s\-.]?)(9?[\s\-.]?[0-9]{4}[\s\-.]?[0-9]{4}))\b')
        n = r2.findall(string)
        n2 = r.findall(string)
        phone_list = []
        for phone in (n + n2):
            phone_list.append(phone[0])
        return phone_list

    def extract_semantic(self, string):
        r = re.compile(r"[^[]*\[([^]]*)\]")
        return r.findall(str(string))

    # Create Tuples of NER Tags
    def listOfTuples(self, l1, l2):
        return list(map(lambda x, y: (x, y), l1, l2))

    # Define NER Function
    def NER(self, note):

        sentences = nltk.sent_tokenize(note)
        org = []
        location = []
        person = []
        misc = []
        
        for sentence in sentences:
            
            results = self.models.ner_predictor.predict(sentence=sentence) # note is the sentence from input
            # results.update(results_sent)
            
            tagged_sent = self.listOfTuples(results["words"], results["tags"])
    
            word = ""
            
            for token, tag in tagged_sent: # token-words in the sent., tag-"B-PER"...
                
                tagged = tag.split("-")[0].strip()
    
                if tag != "O":
                    pos = tag.split("-")[1].strip()
                if tagged == "U":
                    # #print(pos)
                    if pos == "PER" or pos == 'PERSON':
                        person.append(token)
                    elif (pos == "LOC") or (pos == "FAC"):
                        location.append(token)
                    elif pos == "ORG":
                        org.append(token)
                    else:
                        misc.append(token)
                    continue
    
                elif tagged == "B":  # Begin NE
                    word += token + " "
                    continue
    
                elif tagged == "I":  # Inside NE
                    word += token + " "
                    continue
    
                elif tagged == "L":  # Adjacent NE
                    word += token
                    if pos == "PER" or pos == 'PERSON' :
                        person.append(word)
                    elif (pos == "LOC") or (pos == "FAC"):
                        location.append(word)
                    elif pos == "ORG":
                        org.append(word)
                    else:
                        misc.append(word)
    
                    word = ""
                    continue
    
                else:
                    continue

        email = self.extract_email(note)
        domain = self.extract_domain(note)
        phone = self.extract_phone_number(note)
        weight = self.extract_weight(note)
        height = self.extract_height(note)
        ip = self.extract_ip(note)
        financial = self.extract_card_number(note)

        for item in misc:
            if(item in self.models.ethnicity_list):
                self.ethnicity.append(item)
                misc.remove(item)

        # Create person List
        self.person = person

        self.all_entity["business"] = org
        self.all_entity["address"] = location
        self.all_entity["person"] = person
        self.all_entity["misc"] = misc
        self.all_entity["phone"] = phone
        self.all_entity["domain"] = domain
        self.all_entity["email"] = email
        self.all_entity["financial"] = financial
        self.all_entity["ip"] = ip

      
        # Create Named Entity List
        self.Named_entity = self.all_entity["person"] + self.all_entity["business"] + \
            self.all_entity["address"] + self.all_entity["misc"] + self.ethnicity + \
            self.all_entity["email"] + self.all_entity["domain"] + self.all_entity["phone"] + \
            weight + height + self.all_entity["financial"] + self.all_entity["ip"]
        

        return self.Named_entity, self.all_entity
        
        

    def replace_string(self, index1, index2, note_copy, mainstring, replacementstring):
        if index1 == index2:
            mainstring[index1] = replacementstring
        return " ".join(mainstring)

    def replace_coref(self, document, cluster, use_cluster, use_people, note):
        counter = 0
        people_index = 0
        note_copy = note # have a copy with the input sentence
        for node in cluster:
            if(use_cluster[counter]):
                for position in node:
                    note_copy = self.replace_string(
                        position[0], position[1], note_copy, document, use_people[people_index])

                people_index += 1
            counter += 1

        return note_copy

    def useful_clusters(self, cluster, document, namedEntity):
        flag = 0
        useful_cluster = []
        useful_entity = []
        for node in cluster:
            for position in node:
                for person in namedEntity:
                    if(person in ' '.join(document[position[0]:position[1]+1])):
                        useful_entity.append(person)
                        flag = 1
                    if(flag == 1):
                        useful_cluster.append(True)
                        flag = 0
                        break
                    else:
                        useful_cluster.append(False)
                        break
        return useful_cluster, useful_entity

    # Define Coreference function
    def CRF(self, note, namedEntity):

        coref = self.models.coref_predictor.predict(document=note) # copy sentence to document

        document = coref['document']
        cluster = coref['clusters']

        c, e = self.useful_clusters(cluster, document, namedEntity)

        note = self.replace_coref(document, cluster, c, e, note)

        return note

    def extract_sem(self, srl_prediction, Named_entity):

        select_sem = []

        Flag = False

        for verb in srl_prediction['verbs']:

            semantics = self.extract_semantic(verb['description'])

            for arg in semantics:

                # print("=====")
                # print(arg)
                # print("
                select_sem.append(semantics)
                

        return select_sem

    def print_sem(self, select_sem, Named_entity, all_entity):


        antecedent = ""
        r1 = re.compile(r'(1\W*)?([2-9][0-8][0-9])\W*(([2-9][0-9]{2})\W*([0-9]{4}))|(5[0-9][0-9]?\s?)?(\([0-9]{2}\)|[0-9]{2}[\s\-.]?)(9?[\s\-.]?[0-9]{4}[\s\-.]?[0-9]{4})')

        if len(select_sem) > 0:
           
            for sems_array in select_sem:

                for entity in Named_entity:

                    for sem in sems_array:

                        if len(sem.split(": ")) >= 2:
                            
                            if entity in sem.split(": ")[1].strip():
    
                                antecedent = entity
                                
                                break

                    if entity not in self.common_list:
                        if antecedent != "":
                            self.common_list.append(entity)
                            
                       
                        main_rrid = str(uuid.uuid4().int)

                        """create main entity node"""
                        self.relationship = [{
                            "remoteID": main_rrid,
                            "interlinkDirection": "Normal"
                        }]

                        
                        if antecedent in all_entity["person"]:
                            
                            person = {
                                "person": {
                                    "remoteID": main_rrid
                                }
                            }

                            name = HumanName(antecedent)

                            if name['first']:
                                person['person']['given1'] = name['first']

                            if name['middle']:
                                person['person']['given2'] = name['middle']

                            if name['last']:
                                person['person']['surname'] = name['last']

                            self.data.append(person)

                        elif antecedent in all_entity["ip"]:
                            main_rrid = str(uuid.uuid4().int)
                            try:
                                response = DbIpCity.get(antecedent, api_key='free')
                                lon = ""
                                lat = ""
                                if response.longitude is not None:
                                    lon = response.longitude
                                if response.latitude is not None:
                                    lat = response.latitude
                                
                                if ':' in antecedent:
                                                self.data.append({
                                                    "ipAddress": {
                                                        "remoteID": main_rrid,
                                                        "ipAddress": entity,
                                                        "latitude" : str(lat),
                                                        "longitude": str(lon),
                                                        "version6": True
                                                    }
                                                })
                                else:
                                                 self.data.append({
                                                    "ipAddress": {
                                                        "remoteID": main_rrid,
                                                        "ipAddress": entity,
                                                        "latitude" : str(lat),
                                                        "longitude": str(lon)
                                                    }
                                                })

                            except:
                                if ':' in antecedent:
                                    self.data.append({
                                        "ipAddress": {
                                            "remoteID": main_rrid,
                                            "ipAddress": antecedent,
                                            "version6": True
                                        }
                                    })
                                else:
                                    self.data.append({
                                    "ipAddress": {
                                        "remoteID": main_rrid,
                                        "ipAddress": antecedent,
                                    }
                                    })


                        elif antecedent in all_entity["business"]:
                            self.data.append({
                                "business": {
                                    "remoteID": main_rrid,
                                    "businessName": antecedent
                                }
                            })
                        elif antecedent in all_entity["phone"]:
                            phone = r1.findall(antecedent)
                            countryCode = ""
                            if phone[0][0] != "":    
                                countryCode = phone[0][0]
                            elif phone[0][5] != "":   
                                countryCode = phone[0][5]
                            
                            if phone[0][2] != "":
                                self.data.append({
                                    "telephoneNumber": {
                                        "countryCode": countryCode.strip(),
                                        "areaCode": phone[0][1].replace(r'\W', '').strip(),
                                        "number": phone[0][2].strip(),
                                        "remoteID": main_rrid
                                    }
                                 })
                            elif phone[0][7] != "":
                                self.data.append({
                                    "telephoneNumber": {
                                        "countryCode": countryCode.strip(),
                                        "areaCode": phone[0][6].replace(r'\W', '').strip(),
                                        "number": phone[0][7].strip(),
                                        "remoteID": main_rrid
                                    }
                                 })
                            
                        elif antecedent in all_entity["domain"]:
                            
                            self.data.append({
                                "domain": {
                                    "remoteID": main_rrid,
                                    "domainName": antecedent
                                }
                            })
                        elif antecedent in all_entity["email"]:
                            
                            self.data.append({
                                "emailAddress": {
                                    "remoteID": main_rrid,
                                    "emailAddress": antecedent
                                }
                            })

                        elif antecedent in all_entity["address"]:
                            """Parse the address"""
                            try:
#                                addressCA = pyap.parse(antecedent, country='CA')              
#                                addressUS = pyap.parse(antecedent, country='US')
#                                print("address: "+ antecedent)
#                                if len(addressCA) > 0:
#                                    print(addressCA[0].as_dict())
#                                elif len(addressUS) > 0:
#                                    print(addressUS[0].as_dict())
#                                
                                # temp_add, addressType = usaddress.tag(antecedent, self.models.address_parse_dict)
                                # self.map_term(temp_add, "StreetNamePostDirectional", self.models.address_schema_map_direction)
                                # self.map_term(temp_add, "StreetNamePostType", self.models.addressStreetTypeOptions) 
                                # temp_add = json.loads(json.dumps(temp_add))
                                temp_add = idea_address_parser(antecedent)
                            except usaddress.RepeatedLabelError as e :
                                temp_add = {
                                   "notes": "Error parsing \"" + e.parsed_string + "\" from \"" + e.original_string + "\""
                                }
                            temp_add["remoteID"] = main_rrid
                            self.data.append({
                                "address": temp_add
                            })
                                
                        break

                for i in range(1, len(sems_array)):

                    if "ARG" in sems_array[i]:
                        sem_list = sems_array[i].split(": ")[1]

                        for entity in Named_entity:

                            if entity not in self.common_list:

                                self.common_list.append(entity)

                                if (entity in sem_list) or ((antecedent + "==>"+entity) not in self.group_list):
                                    
                                    self.group_list.append(
                                        antecedent + "==>"+entity)

                                    if entity in all_entity["person"]:

                                        """Add relationship Reason"""
                                        self.relationship[0]["reason"] = "Other"
                                        
                                        person = {

                                            "person": {
                                                "remoteID": str(uuid.uuid4().int),
#                                                "relationships": self.relationship
                                            }
                                        }

                                        name = HumanName(entity)

                                        if name['first']:
                                            person['person']['given1'] = name['first']

                                        if name['middle']:
                                            person['person']['maidenName'] = name['middle']

                                        if name['last']:
                                            person['person']['surname'] = name['last']

                                        self.data.append(person)
                                        
                                    if entity in all_entity["business"]:

                                        """Add relationship Reason"""
                                        self.relationship[0]["reason"] = "Other"

                                        self.data.append({

                                            "business": {
                                                "remoteID": str(uuid.uuid4().int),
                                                "businessName": entity,
#                                                "relationships": self.relationship
                                            }
                                        })
                                    elif entity in all_entity["ip"]:
                                        main_rrid = str(uuid.uuid4().int)

                                        try:
                                            response = DbIpCity.get(i, api_key='free')
                                            lon = ""
                                            lat = ""
                                            if response.longitude is not None:
                                                lon = response.longitude
                                            if response.latitude is not None:
                                                lat = response.latitude
                                            
                                            if ':' in entity:
                                                self.data.append({
                                                    "ipAddress": {
                                                        "remoteID": str(uuid.uuid4().int),
                                                        "ipAddress": entity,
                                                        "latitude" : str(lat),
                                                        "longitude": str(lon),
                                                        "version6": True
                                                    }
                                                })
                                            else:
                                                 self.data.append({
                                                    "ipAddress": {
                                                        "remoteID": str(uuid.uuid4().int),
                                                        "ipAddress": entity,
                                                        "latitude" : str(lat),
                                                        "longitude": str(lon)
                                                    }
                                                })
                                        except:
                                            if ':' in entity:
                                                self.data.append({
                                                    "ipAddress": {
                                                        "remoteID": str(uuid.uuid4().int),
                                                        "ipAddress": entity,
                                                        "version6":True
                                                    }
                                                })
                                            else:
                                                self.data.append({
                                                    "ipAddress": {
                                                        "remoteID": str(uuid.uuid4().int),
                                                        "ipAddress": entity,
                                                    }
                                                })

                                    elif entity in all_entity["phone"]:

                                        self.relationship[0]["reason"] = "Current Phone Addresss"

                                        phone = r1.findall(entity)
                                        countryCode = ""
                                        if phone[0][0] != "":    
                                            countryCode = phone[0][0]
                                        elif phone[0][5] != "":   
                                            countryCode = phone[0][5]
                                        
                                        if phone[0][2] != "":
                                            self.data.append({
                                                "telephoneNumber": {
                                                    "countryCode": countryCode.strip(),
                                                    "areaCode": phone[0][1].replace(r'\W', '').strip(),
                                                    "number": phone[0][2].strip(),
                                                    "remoteID": str(uuid.uuid4().int),
                                                }
                                            })
                                        elif phone[0][7] != "":
                                            self.data.append({
                                                "telephoneNumber": {
                                                    "countryCode": countryCode.strip(),
                                                    "areaCode": phone[0][6].replace(r'\W', '').strip(),
                                                    "number": phone[0][7].strip(),
                                                    "remoteID": str(uuid.uuid4().int),
                                                }
                                            })
                                                    
                                    elif entity in all_entity["domain"]:

                                        self.relationship[0]["reason"] = "Other"

                                        self.data.append({
                                            "domain": {
                                                "remoteID": str(uuid.uuid4().int),
                                                "domainName": entity,
#                                                "relationships": self.relationship
                                            }
                                        })
                                    elif entity in all_entity["email"]:

                                        self.relationship[0]["reason"] = "Current Email Address"

                                        self.data.append({
                                            "emailAddress": {
                                                "remoteID": str(uuid.uuid4().int),
                                                "emailAddress": entity,
#                                                "relationships": self.relationship
                                            }
                                        })

                                    elif entity in all_entity["address"]:

                                        #print(self.relationship)

                                        self.relationship[0]["reason"] = "Current Address"

                                        """Parse the address"""
                                        try:
#                                            addressCA = pyap.parse(entity, country='CA')              
#                                            addressUS = pyap.parse(entity, country='US')
#                                            if len(addressCA) > 0:
#                                                print(addressCA[0].as_dict())
#                                            elif len(addressUS) > 0:
#                                                print(addressUS[0].as_dict())
                                            # temp_add, addressType = usaddress.tag(entity, self.models.address_parse_dict)
                                            # self.map_term(temp_add, "StreetNamePostDirectional", self.models.address_schema_map_direction)
                                            # self.map_term(temp_add, "StreetNamePostType", self.models.addressStreetTypeOptions) 
                                            # temp_add = json.loads(json.dumps(temp_add))
                                            temp_add = idea_address_parser(entity)
                                        except usaddress.RepeatedLabelError as e :
                                            temp_add = {
                                               "notes": "Error parsing \"" + e.parsed_string + "\" from \"" + e.original_string + "\""
                                            }
                                        temp_add["remoteID"] = str(uuid.uuid4().int)
#                                        temp_add["relationships"] = self.relationship
                                        self.data.append({
                                            "address": temp_add
                                        })
        else:
            for entity in Named_entity:

                            if entity not in self.common_list:

                                    self.common_list.append(entity)
                                    if entity in all_entity["person"]:

                                        person = {

                                            "person": {
                                                "remoteID": str(uuid.uuid4().int),
#                                                "relationships": self.relationship
                                            }
                                        }

                                        name = HumanName(entity)

                                        if name['first']:
                                            person['person']['given1'] = name['first']

                                        if name['middle']:
                                            person['person']['maidenName'] = name['middle']

                                        if name['last']:
                                            person['person']['surname'] = name['last']

                                        self.data.append(person)
                                        
                                    if entity in all_entity["business"]:

                                        """Add relationship Reason"""
                                        self.data.append({

                                            "business": {
                                                "remoteID": str(uuid.uuid4().int),
                                                "businessName": entity,
#                                                "relationships": self.relationship
                                            }
                                        })
                                    elif entity in all_entity["ip"]:
                                        main_rrid = str(uuid.uuid4().int)

                                        try:
                                            response = DbIpCity.get(i, api_key='free')
                                            lon = ""
                                            lat = ""
                                            if response.longitude is not None:
                                                lon = response.longitude
                                            if response.latitude is not None:
                                                lat = response.latitude
                                            
                                            if ':' in entity:
                                                self.data.append({
                                                    "ipAddress": {
                                                        "remoteID": str(uuid.uuid4().int),
                                                        "ipAddress": entity,
                                                        "latitude" : str(lat),
                                                        "longitude": str(lon),
                                                        "version6": True
                                                    }
                                                })
                                            else:
                                                 self.data.append({
                                                    "ipAddress": {
                                                        "remoteID": str(uuid.uuid4().int),
                                                        "ipAddress": entity,
                                                        "latitude" : str(lat),
                                                        "longitude": str(lon)
                                                    }
                                                })
                                        except:
                                            if ':' in entity:
                                                self.data.append({
                                                    "ipAddress": {
                                                        "remoteID": str(uuid.uuid4().int),
                                                        "ipAddress": entity,
                                                        "version6":True
                                                    }
                                                })
                                            else:
                                                self.data.append({
                                                    "ipAddress": {
                                                        "remoteID": str(uuid.uuid4().int),
                                                        "ipAddress": entity,
                                                    }
                                                })

                                    elif entity in all_entity["phone"]:

                                        phone = r1.findall(entity)
                                        countryCode = ""
                                        if phone[0][0] != "":    
                                            countryCode = phone[0][0]
                                        elif phone[0][5] != "":   
                                            countryCode = phone[0][5]
                                        
                                        if phone[0][2] != "":
                                            self.data.append({
                                                "telephoneNumber": {
                                                    "countryCode": countryCode.strip(),
                                                    "areaCode": phone[0][1].replace(r'\W', '').strip(),
                                                    "number": phone[0][2].strip(),
                                                    "remoteID": str(uuid.uuid4().int),
                                                }
                                            })
                                        elif phone[0][7] != "":
                                            self.data.append({
                                                "telephoneNumber": {
                                                    "countryCode": countryCode.strip(),
                                                    "areaCode": phone[0][6].replace(r'\W', '').strip(),
                                                    "number": phone[0][7].strip(),
                                                    "remoteID": str(uuid.uuid4().int),
                                                }
                                            })
                                                    
                                    elif entity in all_entity["domain"]:


                                        self.data.append({
                                            "domain": {
                                                "remoteID": str(uuid.uuid4().int),
                                                "domainName": entity,
#                                                "relationships": self.relationship
                                            }
                                        })
                                    elif entity in all_entity["email"]:

                                        self.data.append({
                                            "emailAddress": {
                                                "remoteID": str(uuid.uuid4().int),
                                                "emailAddress": entity,
#                                                "relationships": self.relationship
                                            }
                                        })

                                    elif entity in all_entity["address"]:

                                        #print(self.relationship)
                                        """Parse the address"""
                                        try:
#                                            addressCA = pyap.parse(entity, country='CA')              
#                                            addressUS = pyap.parse(entity, country='US')
#                                            if len(addressCA) > 0:
#                                                print(addressCA[0].as_dict())
#                                            elif len(addressUS) > 0:
#                                                print(addressUS[0].as_dict())
                                            # temp_add, addressType = usaddress.tag(entity, self.models.address_parse_dict)
                                            # self.map_term(temp_add, "StreetNamePostDirectional", self.models.address_schema_map_direction)
                                            # self.map_term(temp_add, "StreetNamePostType", self.models.addressStreetTypeOptions) 
                                            # temp_add = json.loads(json.dumps(temp_add))
                                            temp_add = idea_address_parser(entity)
                                        except usaddress.RepeatedLabelError as e :
                                            temp_add = {
                                               "notes": "Error parsing \"" + e.parsed_string + "\" from \"" + e.original_string + "\""
                                            }
                                        temp_add["remoteID"] = str(uuid.uuid4().int)
#                                        temp_add["relationships"] = self.relationship
                                        self.data.append({
                                            "address": temp_add
                                        })

        return self.data

    def exec_sem(self, note, named, all_entity):
        for sentences in sent_tokenize(note):

            srl_prediction = self.models.srl_predictor.predict(sentence=sentences)

            select = self.extract_sem(srl_prediction, named)
           
            group = self.print_sem(select, named, all_entity)

        return group
    
# if __name__ == "__main__":
#     models = Models()
#     ext = Extract(models)
#     print("extracting.....")
#     parse_string ="William Clay ford Jr. is the great grandson of Henry Ford of DEI"
#     # Execute NER Function
#     named, all_entity = ext.NER(parse_string)
    
#     # execute coreference
#     crfText = ext.CRF(parse_string, named)
    
#     # execute Semantic Role Labelling
#     element_array = ext.exec_sem(crfText,named, all_entity)
    
#     print(element_array)
