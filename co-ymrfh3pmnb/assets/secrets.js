/* ------------------------------------------------------------------------
 * Les Chroniques du CO — configuration du tirage
 *
 * CE FICHIER EST GÉNÉRÉ. Ne l'édite pas à la main : lance
 *     python3 tools/tirage.py
 * et remplace ce fichier par ce que le script imprime.
 *
 * Il ne contient aucune adresse e-mail et aucun nom d'élu en clair.
 * - names  : empreinte(sel_noms + adresse) -> prénom, pour saluer le candidat
 * - elect  : empreinte(sel_elus + adresse) des organisateurs désignés
 * Les deux sels étant différents, on ne peut pas rapprocher une entrée de
 * `names` d'une entrée de `elect` sans repartir de l'adresse e-mail.
 *
 * Tirage en cours : saison II, édition I — effectué le 2026-10-27
 * ---------------------------------------------------------------------- */

window.CO_CONFIG = {
  saltNames: 'cd587d72cb2ea30c',
  saltElect: 'e1c0f7510a2327ef',

  names: {
    '5963f96da0ecb62de5119d3a769fc35b8e8874c4': 'Andrea',
    'd74326220810f4ed21441694a82440ac96eda304': 'Zava',
    '471e94c4d1385b318802af25a0fbb0916bb8e5bf': 'Antoine',
    '6361aaa7142564b657e7c860d7366c34e422875a': 'Rémy',
    '8d3df3f209acf4d6df287b08ceb6e8c65d4d5cfb': 'Thibaud',
    'f8a823d765315e3dacbb2ae3f4297c60d4600c59': 'Claire',
    '7d77d4f960a920c1d44f2aa474d9982b53485070': 'Pierre',
    'bb56e7971e39437d1a047cfa213107a24ec92e42': 'Pauline',
    'fccd8f13c4b0dc847e99220b4c75e7eb4b1df050': 'Quentin',
    'ac452560f92ea6d8583f7acb1503011692ac53a5': 'Max',
    '86c28168ad47889a2ea6d04c7747f235d04d43f5': 'Benjamin',
    '0456c7ac5a234a687876fa6fa3dde7f57511fa45': 'Lara'
  },

  elect: [
    '18809d7980073f439de40ff7c8323089dd84d367',
    'ca40db09d00e7bea379d6b06bc6602ce30e46a50'
  ],

  /* Clé de stockage local : change-la à chaque nouvelle édition pour que
     tout le monde puisse rejouer une fois (ex. 's2e2'). */
  edition: 's2e1'
};
