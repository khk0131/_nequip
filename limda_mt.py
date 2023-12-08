"""
各分子が構成する元素の種類と数
"""
CONVERSION_MOLECULE:dict[str,dict[str,int]] = {
    "H2O": {
        'H': 2,
        'O': 1,
        },
    "Fe": {
        'Fe': 1,
        },
    "MoDTC": {
        'C': 6,
        'H': 12,
        'Mo': 2,
        'N': 2,
        'O': 2,
        'S': 6,
        },
    "LIMoDTC": {
        'C': 6,
        'H': 12,
        'Mo': 2,
        'N': 2,
        'O': 2,
        'S': 6,
        },
    "PAO4": {
        'C': 40,
        'H': 82,
        },
    "MoS2": {
        'Mo': 1,
        'S': 2,
        },
    "ZnDTP": {
        'C': 8,
        'H': 20,
        'O': 4,
        'S': 4,
        'P': 2,
        'Zn': 1,
    },
}




#-----------------------------------------------------------------------------
    def zmax_info(self) -> None:
        """
        真空部分に分子を詰める際に、それぞれの元素のz方向に最大値を取るときの座標を取得し、
        セルサイズと照らし合わせながら、詰め方を考察します。
        """
        z_max_coodinate_info = np.zeros(len(self.get_atom_type_set()))
        header_line = ["" for _ in range(len(self.get_atom_type_set()))]
        z_max = ""
        atom_types = set()
  
        for index, row in self.atoms.iterrows():
            if row.type in atom_types:
                if row.z >= z_max_coodinate_info[int(row.type)-1]:
                    z_max = f'{row}\n'
                    z_max_coodinate_info[int(row.type)-1] = row.z
                    header_line[int(row.type)-1] = z_max # z座標が最大のときの、各元素の座標を更新
            else:
                atom_types.add(row.type)
                z_max_coodinate_info[int(row.type)-1] = row.z
                z_max = f'{row}\n'
                header_line[int(row.type)-1] = z_max
            
            if index == self.get_total_atoms() - 1:
                header_line.append('\n')
        
        header_line.insert(0, 'z_max coodinate:\n')
        header_line.append(f'cell size:{self.cell}\n\n')
        header_line.append(f'atom nums:{self.get_total_atoms()}\n\n')
        header_line.append(f'atoms type counts:\n{self.count_atom_types()}\n\n')
        
        with open(f'{os.getcwd()}/ZMAX_INFO', 'w') as ofp:
            ofp.writelines(header_line)
    #---------------------------------------------------------------------------
    def zmin_info(self) -> None:
        """
        真空部分に分子を詰める際に、それぞれの元素のz方向に最小値を取るときの座標を取得し、
        セルサイズと照らし合わせながら、詰め方を考察します。
        """
        z_min_coodinate_info = np.full(len(self.get_atom_type_set()), np.inf)
        header_line = ["" for _ in range(len(self.get_atom_type_set()))]
        z_min = ""
        atom_types = set()
  
        for index, row in self.atoms.iterrows():
            if row.type in atom_types:
                if float(row.z) <= z_min_coodinate_info[int(row.type)-1]:
                    z_min = f'{row}\n'
                    z_min_coodinate_info[int(row.type)-1] = row.z
                    header_line[int(row.type)-1] = z_min # z座標が最小のときの、各元素の座標を更新
            else:
                atom_types.add(row.type)
                z_min_coodinate_info[int(row.type)-1] = row.z
                z_min = f'{row}\n'
                header_line[int(row.type)-1] = z_min
            
            if index == self.get_total_atoms() - 1:
                header_line.append('\n')
        
        header_line.insert(0, 'z_min coodinate:\n')
        header_line.append(f'cell size:{self.cell}\n\n')
        header_line.append(f'atom nums:{self.get_total_atoms()}\n\n')
        header_line.append(f'atoms type counts:\n{self.count_atom_types()}\n\n')
        
        with open(f'{os.getcwd()}/ZMIN_INFO', 'w') as ofp:
            ofp.writelines(header_line)
        
    #------------------------------------------------------------------------------
    
    def type_to_molecules(self, mol_lists: list[str], bond_length: list[list[float]]) -> list[str]:
        """
        各原子がどの分子もしくは原子（単体や金属で存在）するかを判定する。
        各原子の結合している原子をget_neighbor_listで取得後、bfsを行って、どの分子に所属しているかを判定する。
        -argument-
        mol_lists: 使用する分子もしくは、金属原子のlist[str]
        """
        actual_mol_symbols = ["" for _ in range(len(self))] # idにおける分子の種類を返す
        actual_mol_dicts = {}
        for mol_list in mol_lists:
            actual_mol_dicts[mol_list] = C.CONVERSION_MOLECULE[mol_list]
        
        self.atoms = self.atoms.sort_values('type').reset_index(drop=True)
        neighbor_lists = self.get_neighbor_list(bond_length=bond_length)
        neighbor_bool = [False for _ in range(len(self))]
        
        queue = deque()
        
        for i in range(len(self)):
            if all(neighbor_bool):
                break
            if neighbor_bool[i]:
                continue
            queue.append(i)
            mol_dict = {} # bfsしたこの分子を構成する原子のsymbolと個数を記録 -> 後で上のdictと照らし合わせてどの分子か判定
            atom_id_count = [] # bfsした分子のidを記録
            atom_type_set = set() # bfsした分子に出てきた原子のtypeを記録
            while queue:
                current_atom = queue.popleft()
                if neighbor_bool[current_atom]:
                    continue
                neighbor_bool[current_atom] = True
                atom_id_count.append(current_atom)

                if self.atoms.loc[current_atom, 'type'] not in atom_type_set:
                    mol_dict.setdefault(self.atom_type_to_symbol[self.atoms.loc[current_atom, 'type']], 1)
                else:
                    mol_dict[self.atom_type_to_symbol[self.atoms.loc[current_atom, 'type']]] += 1
                
                atom_type_set.add(self.atoms.loc[current_atom, 'type'])
                next_neighbor_lists = neighbor_lists[current_atom]
                for next_atom_id in range(len(next_neighbor_lists)):
                    if neighbor_bool[next_neighbor_lists[next_atom_id]] == False:
                        queue.append(next_neighbor_lists[next_atom_id])
            
            mol_dict = dict(sorted(mol_dict.items()))
            
            #-- MoS2かの判定
            count: int = 0
            for key, values in mol_dict.items():
                if (key == 'Mo' and values%2 == 0) or (key == 'S' and values%2 == 0):
                    count += 1
            if count == len(mol_dict):
                mol_dict = {
                    'Mo': 1,
                    'S': 2,
                }
            #--
            mol_symbol =  [key for key, value in actual_mol_dicts.items() if value == mol_dict][0]

            for j in range(len(atom_id_count)):
                actual_mol_symbols[atom_id_count[j]] = mol_symbol
                
        return actual_mol_symbols
    
    #-----------------------------------------------------------------------------------
    
    def initial_velocity(self, mol_lists: list[str], target: str, molecular_velocity: dict[str, float], bond_length: list[list[float]], slab_symbol: str="Fe") -> dict[str, float]:
        """
        構成する分子もしくは金属原子の種類の初期速度を、系の重心が移動しないような各分子もしくは金属原子の運動量から求める。
        
        self.molecular_num: self.type_to_molecules()で各原子が所属している分子のlistを作成した後、dictへ変換。self.molecular_numとして保持
        self.molecular_weight: 分子1個（もしくは、金属原子1個）の質量を計算し、保持する
        
        -argument-
        mol_lists: 使用する分子もしくは、金属原子のlist[str]
        target: どの分子に主に初期速度を与えたいか
        molecular_velocity: targetの初期速度, target以外の分子もしくは金属原子の初期速度を与える。正し、与える種類数は (系内に存在する種類 - 1) である。残り1つは、系の重心が移動しないような運動量の式で求める。
        """
        momentum_except_target = 0.0
        molecular_symbol_num = Counter(self.type_to_molecules(mol_lists, bond_length))
        for mol_symbol, atom_num in molecular_symbol_num.items():
            mol_dict = C.CONVERSION_MOLECULE[mol_symbol]
            molecular_symbol_num[mol_symbol] /= sum(mol_dict.values())
      
        self.molecular_num = molecular_symbol_num 
        
        mol_weight = {}
        for mol_symbol in mol_lists:
            mol_dict = C.CONVERSION_MOLECULE[mol_symbol]
            weights = 0
            for atom_symbol, atom_num in mol_dict.items():
                weights += C.ATOM_SYMBOL_TO_MASS[atom_symbol] * atom_num / C.AVOGADORO_CONST
            mol_weight[mol_symbol] = weights
        self.molecular_weight = mol_weight
        target_momentum = self.molecular_num[target] * self.molecular_weight[target] * molecular_velocity[target]
        
        #------- target分子を１個だけに限定する
        #---------------------------------
        for mol_symbol in mol_lists:
            if mol_symbol == target or mol_symbol == slab_symbol:
                continue
            else:
                momentum_except_target += self.molecular_num[mol_symbol] * self.molecular_weight[mol_symbol] * molecular_velocity[mol_symbol]  

        slab_velocity = - (target_momentum + momentum_except_target) / (self.molecular_num[slab_symbol] * self.molecular_weight[slab_symbol])
        
        molecular_velocity[slab_symbol] = slab_velocity

        return molecular_velocity
        