-- SPJ Collector — Seed data: ~300 high-priority words for AI training
-- Organized into 15 themes of ~20 words each
-- Priority: everyday vocabulary that the AI needs most (underrepresented in current training data)
SET NAMES utf8mb4;

-- ═══════════════════════════════════════
-- THEMES
-- ═══════════════════════════════════════

INSERT INTO themes (name, emoji, sort_order) VALUES
('Pozdravy a základné', '👋', 1),
('Rodina a ľudia', '👨‍👩‍👧‍👦', 2),
('Jedlo a nápoje', '🍞', 3),
('Farby', '🎨', 4),
('Čísla a počítanie', '🔢', 5),
('Čas a dni', '📅', 6),
('Škola a vzdelávanie', '🎓', 7),
('Emócie a pocity', '😊', 8),
('Zvieratá', '🐕', 9),
('Dom a miesta', '🏠', 10),
('Doprava a cestovanie', '🚗', 11),
('Telo a zdravie', '🏥', 12),
('Práca a povolania', '💼', 13),
('Príroda a počasie', '🌤️', 14),
('Slovesá — bežné', '🏃', 15);

-- ═══════════════════════════════════════
-- SIGNS (words) — grouped by theme
-- ═══════════════════════════════════════

-- Theme 1: Pozdravy a základné (20 words)
INSERT INTO signs (gloss_id, word_sk, theme_id, sort_order_in_theme) VALUES
('AHOJ-1', 'ahoj', 1, 1),
('DOBRY-DEN-1', 'dobrý deň', 1, 2),
('DOVIDENIA-1', 'dovidenia', 1, 3),
('DAKUJEM-1', 'ďakujem', 1, 4),
('PROSIM-1', 'prosím', 1, 5),
('ANO-1', 'áno', 1, 6),
('NIE-1', 'nie', 1, 7),
('DOBRY-RANO-1', 'dobré ráno', 1, 8),
('DOBRU-NOC-1', 'dobrú noc', 1, 9),
('PREPACTE-1', 'prepáčte', 1, 10),
('NEVIEM-1', 'neviem', 1, 11),
('ROZUMIEM-1', 'rozumiem', 1, 12),
('NEROZUMIEM-1', 'nerozumiem', 1, 13),
('POMOC-1', 'pomoc', 1, 14),
('KDE-1', 'kde', 1, 15),
('KTO-1', 'kto', 1, 16),
('CO-1', 'čo', 1, 17),
('KEDY-1', 'kedy', 1, 18),
('AKO-1', 'ako', 1, 19),
('PRECO-1', 'prečo', 1, 20);

-- Theme 2: Rodina a ľudia (20 words)
INSERT INTO signs (gloss_id, word_sk, theme_id, sort_order_in_theme) VALUES
('MAMA-1', 'mama', 2, 1),
('OTEC-1', 'otec', 2, 2),
('BRAT-1', 'brat', 2, 3),
('SESTRA-1', 'sestra', 2, 4),
('BABKA-1', 'babka', 2, 5),
('DEDKO-1', 'dedko', 2, 6),
('DIETA-1', 'dieťa', 2, 7),
('RODINA-1', 'rodina', 2, 8),
('PRIATEL-1', 'priateľ', 2, 9),
('PRIATEKA-1', 'priateľka', 2, 10),
('MANZEL-1', 'manžel', 2, 11),
('MANZELKA-1', 'manželka', 2, 12),
('CLOVEK-1', 'človek', 2, 13),
('CHLAPEC-1', 'chlapec', 2, 14),
('DIEVCA-1', 'dievča', 2, 15),
('UCITEL-1', 'učiteľ', 2, 16),
('ZIAK-1', 'žiak', 2, 17),
('SUSED-1', 'sused', 2, 18),
('NEPOCUJUCI-1', 'nepočujúci', 2, 19),
('POCUJUCI-1', 'počujúci', 2, 20);

-- Theme 3: Jedlo a nápoje (20 words)
INSERT INTO signs (gloss_id, word_sk, theme_id, sort_order_in_theme) VALUES
('VODA-1', 'voda', 3, 1),
('CHLIEB-1', 'chlieb', 3, 2),
('MLIEKO-1', 'mlieko', 3, 3),
('KAVA-1', 'káva', 3, 4),
('CAJ-1', 'čaj', 3, 5),
('JABLKO-1', 'jablko', 3, 6),
('MASO-1', 'mäso', 3, 7),
('SYR-1', 'syr', 3, 8),
('RYBA-1', 'ryba', 3, 9),
('ZELENINA-1', 'zelenina', 3, 10),
('OVOCIE-1', 'ovocie', 3, 11),
('POLIEVKA-1', 'polievka', 3, 12),
('JEDLO-1', 'jedlo', 3, 13),
('JEST-1', 'jesť', 3, 14),
('PIT-1', 'piť', 3, 15),
('CUKR-1', 'cukor', 3, 16),
('SOL-1', 'soľ', 3, 17),
('MASLO-1', 'maslo', 3, 18),
('ZMRZLINA-1', 'zmrzlina', 3, 19),
('COKOLADA-1', 'čokoláda', 3, 20);

-- Theme 4: Farby (15 words)
INSERT INTO signs (gloss_id, word_sk, theme_id, sort_order_in_theme) VALUES
('CERVENA-1', 'červená', 4, 1),
('MODRA-1', 'modrá', 4, 2),
('ZELENA-1', 'zelená', 4, 3),
('ZLTA-1', 'žltá', 4, 4),
('BIELA-1', 'biela', 4, 5),
('CIERNA-1', 'čierna', 4, 6),
('ORANZOVA-1', 'oranžová', 4, 7),
('RUZOVA-1', 'ružová', 4, 8),
('HNEDA-1', 'hnedá', 4, 9),
('FIALOVA-1', 'fialová', 4, 10),
('SEDA-1', 'šedá', 4, 11),
('ZLATA-1', 'zlatá', 4, 12),
('STRIEBORNÁ-1', 'strieborná', 4, 13),
('SVETLA-1', 'svetlá', 4, 14),
('TMAVA-1', 'tmavá', 4, 15);

-- Theme 5: Čísla a počítanie (20 words)
INSERT INTO signs (gloss_id, word_sk, theme_id, sort_order_in_theme) VALUES
('JEDEN-1', 'jeden', 5, 1),
('DVA-1', 'dva', 5, 2),
('TRI-1', 'tri', 5, 3),
('STYRI-1', 'štyri', 5, 4),
('PAT-1', 'päť', 5, 5),
('SEST-1', 'šesť', 5, 6),
('SEDEM-1', 'sedem', 5, 7),
('OSEM-1', 'osem', 5, 8),
('DEVAT-1', 'deväť', 5, 9),
('DESAT-1', 'desať', 5, 10),
('STO-1', 'sto', 5, 11),
('TISIC-1', 'tisíc', 5, 12),
('PRVY-1', 'prvý', 5, 13),
('POSLEDNY-1', 'posledný', 5, 14),
('VELA-1', 'veľa', 5, 15),
('MALO-1', 'málo', 5, 16),
('VSETKO-1', 'všetko', 5, 17),
('NIC-1', 'nič', 5, 18),
('KOLKO-1', 'koľko', 5, 19),
('KAZDY-1', 'každý', 5, 20);

-- Theme 6: Čas a dni (20 words)
INSERT INTO signs (gloss_id, word_sk, theme_id, sort_order_in_theme) VALUES
('DNES-1', 'dnes', 6, 1),
('VCERA-1', 'včera', 6, 2),
('ZAJTRA-1', 'zajtra', 6, 3),
('TERAZ-1', 'teraz', 6, 4),
('POTOM-1', 'potom', 6, 5),
('PONDELOK-1', 'pondelok', 6, 6),
('UTOROK-1', 'utorok', 6, 7),
('STREDA-1', 'streda', 6, 8),
('STVRTOK-1', 'štvrtok', 6, 9),
('PIATOK-1', 'piatok', 6, 10),
('SOBOTA-1', 'sobota', 6, 11),
('NEDELA-1', 'nedeľa', 6, 12),
('RANO-1', 'ráno', 6, 13),
('VECER-1', 'večer', 6, 14),
('HODINA-1', 'hodina', 6, 15),
('MINUTA-1', 'minúta', 6, 16),
('DEN-1', 'deň', 6, 17),
('TYZDEN-1', 'týždeň', 6, 18),
('MESIAC-1', 'mesiac', 6, 19),
('ROK-1', 'rok', 6, 20);

-- Theme 7: Škola a vzdelávanie (20 words)
INSERT INTO signs (gloss_id, word_sk, theme_id, sort_order_in_theme) VALUES
('SKOLA-1', 'škola', 7, 1),
('TRIEDA-1', 'trieda', 7, 2),
('KNIHA-1', 'kniha', 7, 3),
('UCIT-SA-1', 'učiť sa', 7, 4),
('PISAT-1', 'písať', 7, 5),
('CITAT-1', 'čítať', 7, 6),
('POCITAT-1', 'počítač', 7, 7),
('PERO-1', 'pero', 7, 8),
('PAPIER-1', 'papier', 7, 9),
('SKUSKA-1', 'skúška', 7, 10),
('ULOHA-1', 'úloha', 7, 11),
('PRAZDNINY-1', 'prázdniny', 7, 12),
('UNIVERZITA-1', 'univerzita', 7, 13),
('STUDENT-1', 'študent', 7, 14),
('VYUCBA-1', 'výučba', 7, 15),
('TLMOCNIK-1', 'tlmočník', 7, 16),
('POSUNKOVY-JAZYK-1', 'posunkový jazyk', 7, 17),
('SLOVNIK-1', 'slovník', 7, 18),
('POZNAMKA-1', 'poznámka', 7, 19),
('VYSVEDCENIE-1', 'vysvedčenie', 7, 20);

-- Theme 8: Emócie a pocity (20 words)
INSERT INTO signs (gloss_id, word_sk, theme_id, sort_order_in_theme) VALUES
('STASTNY-1', 'šťastný', 8, 1),
('SMUTNY-1', 'smutný', 8, 2),
('NAHNEVANY-1', 'nahnevaný', 8, 3),
('BAT-SA-1', 'báť sa', 8, 4),
('LASKA-1', 'láska', 8, 5),
('PREKVAPENY-1', 'prekvapený', 8, 6),
('UNAVENY-1', 'unavený', 8, 7),
('SPOKOJNY-1', 'spokojný', 8, 8),
('NUDA-1', 'nuda', 8, 9),
('STAROST-1', 'starosť', 8, 10),
('NADEJ-1', 'nádej', 8, 11),
('SMIAT-SA-1', 'smiať sa', 8, 12),
('PLAKAT-1', 'plakať', 8, 13),
('HANBIT-SA-1', 'hanbiť sa', 8, 14),
('ZAVIDIET-1', 'závidieť', 8, 15),
('DOVEROVAT-1', 'dôverovať', 8, 16),
('PACAT-SA-1', 'páčiť sa', 8, 17),
('NEPACAT-SA-1', 'nepáčiť sa', 8, 18),
('NENAVIDIET-1', 'nenávidieť', 8, 19),
('HRDÝ-1', 'hrdý', 8, 20);

-- Theme 9: Zvieratá (20 words)
INSERT INTO signs (gloss_id, word_sk, theme_id, sort_order_in_theme) VALUES
('PES-1', 'pes', 9, 1),
('MACKA-1', 'mačka', 9, 2),
('VTAK-1', 'vták', 9, 3),
('KON-1', 'kôň', 9, 4),
('KRAVA-1', 'krava', 9, 5),
('PRASIATKO-1', 'prasa', 9, 6),
('ZAJAC-1', 'zajac', 9, 7),
('MEDVED-1', 'medveď', 9, 8),
('SLON-1', 'slon', 9, 9),
('LEV-1', 'lev', 9, 10),
('MOTYL-1', 'motýľ', 9, 11),
('HADIK-1', 'had', 9, 12),
('ZABA-1', 'žaba', 9, 13),
('MYSKA-1', 'myš', 9, 14),
('OVCA-1', 'ovca', 9, 15),
('KOZA-1', 'koza', 9, 16),
('SLIEPKA-1', 'sliepka', 9, 17),
('KOHUT-1', 'kohút', 9, 18),
('OPICA-1', 'opica', 9, 19),
('VLKA-1', 'vlk', 9, 20);

-- Theme 10: Dom a miesta (20 words)
INSERT INTO signs (gloss_id, word_sk, theme_id, sort_order_in_theme) VALUES
('DOM-1', 'dom', 10, 1),
('BYT-1', 'byt', 10, 2),
('IZBA-1', 'izba', 10, 3),
('KUCHYNA-1', 'kuchyňa', 10, 4),
('KUPELNA-1', 'kúpeľňa', 10, 5),
('OBCHOD-1', 'obchod', 10, 6),
('NEMOCNICA-1', 'nemocnica', 10, 7),
('KOSTOL-1', 'kostol', 10, 8),
('MESTO-1', 'mesto', 10, 9),
('DEDINA-1', 'dedina', 10, 10),
('ULICA-1', 'ulica', 10, 11),
('PARK-1', 'park', 10, 12),
('RESTAURACIA-1', 'reštaurácia', 10, 13),
('HOTEL-1', 'hotel', 10, 14),
('LETISKO-1', 'letisko', 10, 15),
('STANICA-1', 'stanica', 10, 16),
('KNIZNICA-1', 'knižnica', 10, 17),
('DIVADLO-1', 'divadlo', 10, 18),
('BRATISLAVA-1', 'Bratislava', 10, 19),
('SLOVENSKO-1', 'Slovensko', 10, 20);

-- Theme 11: Doprava a cestovanie (18 words)
INSERT INTO signs (gloss_id, word_sk, theme_id, sort_order_in_theme) VALUES
('AUTO-1', 'auto', 11, 1),
('AUTOBUS-1', 'autobus', 11, 2),
('VLAK-1', 'vlak', 11, 3),
('LIETADLO-1', 'lietadlo', 11, 4),
('BICYKEL-1', 'bicykel', 11, 5),
('LOD-1', 'loď', 11, 6),
('TRAMVAJ-1', 'električka', 11, 7),
('CESTOVAT-1', 'cestovať', 11, 8),
('LISTOK-1', 'lístok', 11, 9),
('ZASTAVKA-1', 'zastávka', 11, 10),
('PRAVO-1', 'vpravo', 11, 11),
('LAVO-1', 'vľavo', 11, 12),
('ROVNO-1', 'rovno', 11, 13),
('DALEKO-1', 'ďaleko', 11, 14),
('BLIZKO-1', 'blízko', 11, 15),
('HORE-1', 'hore', 11, 16),
('DOLE-1', 'dole', 11, 17),
('MAPA-1', 'mapa', 11, 18);

-- Theme 12: Telo a zdravie (20 words)
INSERT INTO signs (gloss_id, word_sk, theme_id, sort_order_in_theme) VALUES
('HLAVA-1', 'hlava', 12, 1),
('RUKA-1', 'ruka', 12, 2),
('NOHA-1', 'noha', 12, 3),
('OKO-1', 'oko', 12, 4),
('UCHO-1', 'ucho', 12, 5),
('NOS-1', 'nos', 12, 6),
('USTA-1', 'ústa', 12, 7),
('SRDCE-1', 'srdce', 12, 8),
('CHORY-1', 'chorý', 12, 9),
('ZDRAVY-1', 'zdravý', 12, 10),
('LEKAR-1', 'lekár', 12, 11),
('LIEK-1', 'liek', 12, 12),
('BOLET-1', 'bolieť', 12, 13),
('TEPLOTA-1', 'teplota', 12, 14),
('OPERACIA-1', 'operácia', 12, 15),
('NACHLAD-1', 'nachladnutie', 12, 16),
('ALERGIA-1', 'alergia', 12, 17),
('ZUBAR-1', 'zubár', 12, 18),
('NEPOCUJUCI-2', 'hluchý', 12, 19),
('NACUCI-PRISTROJ-1', 'načúvací prístroj', 12, 20);

-- Theme 13: Práca a povolania (20 words)
INSERT INTO signs (gloss_id, word_sk, theme_id, sort_order_in_theme) VALUES
('PRACA-1', 'práca', 13, 1),
('PRACOVAT-1', 'pracovať', 13, 2),
('KANCELARIA-1', 'kancelária', 13, 3),
('POCITAC-1', 'počítač', 13, 4),
('TELEFON-1', 'telefón', 13, 5),
('EMAIL-1', 'email', 13, 6),
('STRETNUTIE-1', 'stretnutie', 13, 7),
('SEF-1', 'šéf', 13, 8),
('KOLEGA-1', 'kolega', 13, 9),
('PLAT-1', 'plat', 13, 10),
('POLICAJT-1', 'policajt', 13, 11),
('HASIC-1', 'hasič', 13, 12),
('KUCHAR-1', 'kuchár', 13, 13),
('VODIC-1', 'vodič', 13, 14),
('PREDAVAC-1', 'predavač', 13, 15),
('PROGRAMATOR-1', 'programátor', 13, 16),
('UMELEC-1', 'umelec', 13, 17),
('SPORTOVEC-1', 'športovec', 13, 18),
('DOVOLENKLA-1', 'dovolenka', 13, 19),
('PENIZE-1', 'peniaze', 13, 20);

-- Theme 14: Príroda a počasie (18 words)
INSERT INTO signs (gloss_id, word_sk, theme_id, sort_order_in_theme) VALUES
('SLNKO-1', 'slnko', 14, 1),
('DAZD-1', 'dážď', 14, 2),
('SNEH-1', 'sneh', 14, 3),
('VIETOR-1', 'vietor', 14, 4),
('OBLAK-1', 'oblak', 14, 5),
('TEPLO-1', 'teplo', 14, 6),
('ZIMA-1', 'zima', 14, 7),
('STROM-1', 'strom', 14, 8),
('KVET-1', 'kvet', 14, 9),
('HORA-1', 'hora', 14, 10),
('RIEKA-1', 'rieka', 14, 11),
('MORE-1', 'more', 14, 12),
('LES-1', 'les', 14, 13),
('JAR-1', 'jar', 14, 14),
('LETO-1', 'leto', 14, 15),
('JESEN-1', 'jeseň', 14, 16),
('ZIMAROCNE-1', 'zima (ročné)', 14, 17),
('HVIEZDA-1', 'hviezda', 14, 18);

-- Theme 15: Slovesá — bežné (20 words)
INSERT INTO signs (gloss_id, word_sk, theme_id, sort_order_in_theme) VALUES
('IST-1', 'ísť', 15, 1),
('ROBIT-1', 'robiť', 15, 2),
('CHCIET-1', 'chcieť', 15, 3),
('MOCT-1', 'môcť', 15, 4),
('VEDIET-1', 'vedieť', 15, 5),
('VIDIET-1', 'vidieť', 15, 6),
('POCUT-1', 'počuť', 15, 7),
('HOVORIT-1', 'hovoriť', 15, 8),
('POSUNKOVAT-1', 'posunkovať', 15, 9),
('SPAT-1', 'spať', 15, 10),
('KUPIT-1', 'kúpiť', 15, 11),
('DAT-1', 'dať', 15, 12),
('VZIAT-1', 'vziať', 15, 13),
('OTVORIT-1', 'otvoriť', 15, 14),
('ZATVORIT-1', 'zatvoriť', 15, 15),
('HLADAT-1', 'hľadať', 15, 16),
('CEKAT-1', 'čakať', 15, 17),
('POMAHAAT-1', 'pomáhať', 15, 18),
('ZAVOLAT-1', 'zavolať', 15, 19),
('MYSLIET-1', 'myslieť', 15, 20);
