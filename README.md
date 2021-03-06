# Stencil in MPI
## Compiler le projet

### Version de base
```sh
make
```

Pour la version mélangeant mise-à-jour et test de convergence :

```sh
make stencil_all_in_one
```
### Version OpenMP

```sh
make stencil_omp
```

### Version MPI

```sh
make stencil_mpi
```

### Version MPI + OpenMP

```sh
make stencil_mpi_omp
```
## Exécution

Chaque règle `make` créé un exécutable du même nom.
La version de base porte le nom de **stencil**.

### Version de base
Pour la version de base, et *all_in_one*, il faut préciser un argument -s donnant la taille maximale du problème que l'on veut tester.
Exemple :

```sh
./stencil -s 1000
```

va tester les tailles de 60 à 1000 en augmentant la taille de 25% (on n'atteint pas forcément la taille maximale indiquée).
La taille de 60 comme minimum est une constante, elle peut être modifiée dans le code pour être plus petite.

La sortie est au format *csv*.

### Version MPI

Pour les tests de scalabilité forte et faible deux scripts permettent de lancer les tests sur plafrim.
Ils vont chargés des modules si non déjà chargés.
Ils prennent une taille en argument.

Pour la scalabilité forte il s'agit de **N**, soit un côté de la matrice globale. (strong_scaling.sh)

Pour la scalabilité faible il s'agit de **n**, soit un côté de la matrice locale. (weak_scaling.sh)

Les programmes génèrent des fchiers scala-weak-**size**.csv (scala-strong-**size**.csv), où **size** est la valeur de l'agument passé au programme, au format *csv*.
