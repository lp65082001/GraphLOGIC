package require psfgen   
resetpsf
#topology KKdeH_LNLcharge.txt
topology top_all22_prot_HYP_HYL.rtf
pdbalias atom ILE CD1 CD 
pdbalias residue HIS HSE

segment  A {pdb homo_A.pdb}
#segment  0B {pdb homo_0B.pdb}
segment  B {pdb homo_B.pdb}
segment  C {pdb homo_C.pdb}



#patch deH_LNL B:1057 B:1996
#patch deH_LNL C:2089 C:3026
#patch deH_LNL A:103 A:1046
#patch deH_LNL B:1150 C:3126
#patch deH_LNL 1B:1057 1B:1996
#patch deH_LNL 1C:2089 1C:3026
#patch deH_LNL 1A:103 1A:1046
#patch deH_LNL 1B:1150 1C:3126
# Read protein coordinates from PDB file
# formerly "alias atom ..."
regenerate angles dihedrals

pdbalias atom ILE CD1 CD
coordpdb homo_A.pdb A
coordpdb homo_B.pdb B
coordpdb homo_C.pdb C



guesscoord	 

writepdb homo_initial.pdb	 
writepsf homo_initial.psf