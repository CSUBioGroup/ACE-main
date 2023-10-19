library(StabMap)
library(glue)
library(scran)
library(Matrix)
library(rlang)
library(Seurat)

set.seed(2021)

data_dir = '/home/cb213/local/cache3/yxh/Data/seurat4-CITE-reference/pbmc_multimodal_2023.rds'
data_dir2= '/home/yxh/gitrepo/multi-omics-matching/Visualizations/time_complx/inputs/'

cite = readRDS(data_dir)
variable_rna_features = VariableFeatures(cite, assay='SCT')
variable_adt_features = VariableFeatures(cite, assay='ADT')

rna_norm = cite[['SCT']]@data[variable_rna_features,]
adt_norm = cite[['ADT']]@data[variable_adt_features,]
  
for (rate in c(0.01, 0.1, 0.2, 0.4, 0.8, 1.0)){
  smp_names = read.table(paste0(data_dir2, glue('names_{rate}.csv')), header=T, sep=',')$X0
  rna_norm_subset = rna_norm[, smp_names]
  adt_norm_subset = adt_norm[, smp_names]
  n_smp = length(smp_names)
  n_interval = as.integer(n_smp / 3)

  mult_lognorm = rbind(
    rna_norm_subset[, 1:n_interval], 
    adt_norm_subset[, 1:n_interval]  # (n_mult+1):(n_mult+n_single)
  )  
  # dim(mult_lognorm)

  # stabmap !!!!!!!!!!!!!!
  single_rna_lognorm = rna_norm_subset[, (n_interval+1):(2*n_interval)]
  single_adt_lognorm = adt_norm_subset[, (2*n_interval+1):(3*n_interval)]
  colnames(single_rna_lognorm) = paste0('rna_', colnames(single_rna_lognorm))
  colnames(single_adt_lognorm) = paste0('adt_', colnames(single_adt_lognorm))
  assay_list_indirect = list(
    RNA = as(single_rna_lognorm, 'dgCMatrix'),    # dgCMatrix
    Multiome = as(mult_lognorm, 'dgCMatrix'), # dgCMatrix
    ADT = as(single_adt_lognorm, 'dgCMatrix')  # dgCMatrix
  )

  # lapply(assay_list_indirect, dim)

  # jpeg(file="/home/yxh/gitrepo/multi-omics-matching/StabMap-main/vignettes/CITE-mosaic-overlapFeature.jpg", 
  #   width = 4, height = 4, units = 'in', res = 300)
  mosaicDataUpSet(assay_list_indirect, plot = FALSE)
  # dev.off()

  # jpeg(file="/home/yxh/gitrepo/multi-omics-matching/StabMap-main/vignettes/CITE-mosaic-topology.jpg", 
  #   width = 4, height = 4, units = 'in', res = 300)
  mdt_indirect = mosaicDataTopology(assay_list_indirect)
  # plot(mdt_indirect)
  # dev.off()
  start.time = Sys.time()
  stab_indirect = stabMap(assay_list_indirect,
                          reference_list = c("Multiome"),
                          maxFeatures=5000,
                          plot = FALSE)
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  print('===================================')
  print(rate)
  print(time.taken)
  print('===================================')

}

# [1] "==================================="
# [1] 0.01
# Time difference of 1.994172 secs
# [1] "==================================="
# treating "Multiome" as reference
# generating embedding for path with reference "Multiome": "Multiome"
# generating embedding for path with reference "Multiome": "RNA" -> "Multiome"
# generating embedding for path with reference "Multiome": "ADT" -> "Multiome"
# [1] "==================================="
# [1] 0.1
# Time difference of 34.75104 secs
# [1] "==================================="
# treating "Multiome" as reference
# generating embedding for path with reference "Multiome": "Multiome"
# generating embedding for path with reference "Multiome": "RNA" -> "Multiome"
# generating embedding for path with reference "Multiome": "ADT" -> "Multiome"
# [1] "==================================="
# [1] 0.2
# Time difference of 7.062478 mins
# [1] "==================================="
# treating "Multiome" as reference
# generating embedding for path with reference "Multiome": "Multiome"
# generating embedding for path with reference "Multiome": "RNA" -> "Multiome"
# generating embedding for path with reference "Multiome": "ADT" -> "Multiome"
# [1] "==================================="
# [1] 0.4
# Time difference of 15.38837 mins
# [1] "==================================="
# treating "Multiome" as reference
# generating embedding for path with reference "Multiome": "Multiome"
# generating embedding for path with reference "Multiome": "RNA" -> "Multiome"
# generating embedding for path with reference "Multiome": "ADT" -> "Multiome"
# [1] "==================================="
# [1] 0.8
# Time difference of 16.2929 mins
# [1] "==================================="

# [1] "==================================="
# [1] 1
# Time difference of 17.63572 mins
# [1] "==================================="
