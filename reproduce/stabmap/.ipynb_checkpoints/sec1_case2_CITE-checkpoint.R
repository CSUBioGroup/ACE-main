library(StabMap)
# library(SingleCellMultiModal)
library(scran)
library(Matrix)

set.seed(2021)

data_dir = '/home/yxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/datasets/match_modality/openproblems_bmmc_cite_phase2_mod2/'

# loading 
mult_rna_lognorm = t(Matrix::readMM(paste0(data_dir, 'gex_hvg_train.mtx')))
mult_adt_lognorm = t(Matrix::readMM(paste0(data_dir, 'adt_train.mtx')))
single_rna_lognorm = t(Matrix::readMM(paste0(data_dir, 'gex_hvg_test.mtx')))
single_adt_lognorm = t(Matrix::readMM(paste0(data_dir, 'adt_test.mtx')))

hvg = read.table(file=paste0(data_dir, 'gex_hvg.csv'), sep=',', header=T)
adt_var = read.table(file=paste0(data_dir, 'adt_var_names.csv'), sep=',', header=T)
mult_names = read.table(file=paste0(data_dir, 'gex_train_obs_names.csv'), sep=',', header=T)
single_rna_names = read.table(file=paste0(data_dir, 'gex_test_obs_names.csv'), sep=',', header=T)
single_adt_names = read.table(file=paste0(data_dir, 'adt_test_obs_names.csv'), sep=',', header=T)

rownames(mult_rna_lognorm) = hvg[['X0']]
rownames(single_rna_lognorm) = hvg[['X0']]
rownames(mult_adt_lognorm) = adt_var[['X0']]
rownames(single_adt_lognorm) = adt_var[['X0']]

colnames(mult_rna_lognorm) = mult_names[['X0']]
colnames(mult_adt_lognorm) = mult_names[['X0']]
colnames(single_rna_lognorm) = single_rna_names[['X0']]
colnames(single_adt_lognorm) = single_adt_names[['X0']]

mult_lognorm = rbind(mult_rna_lognorm, mult_adt_lognorm)  # logcount(): dgCMatrix
dim(mult_lognorm)


# stabmap !!!!!!!!!!!!!!
assay_list_indirect = list(
  RNA = as(single_rna_lognorm, 'dgCMatrix'),    # dgCMatrix
  Multiome = mult_lognorm, # dgCMatrix
  ADT = as(single_adt_lognorm, 'dgCMatrix')   # dgCMatrix
)

lapply(assay_list_indirect, dim)

jpeg(file="/home/yxh/gitrepo/multi-omics-matching/StabMap-main/vignettes/CITE-mosaic-overlapFeature.jpg", 
  width = 4, height = 4, units = 'in', res = 300)
mosaicDataUpSet(assay_list_indirect, plot = FALSE)
dev.off()

jpeg(file="/home/yxh/gitrepo/multi-omics-matching/StabMap-main/vignettes/CITE-mosaic-topology.jpg", 
  width = 4, height = 4, units = 'in', res = 300)
mdt_indirect = mosaicDataTopology(assay_list_indirect)
plot(mdt_indirect)
dev.off()

# Running !!!!!!!!!!
# return matrix array
stab_indirect = stabMap(assay_list_indirect,
                        reference_list = c("Multiome"),
                        maxFeatures=6000,
                        plot = FALSE)


out_dir = '/home/yxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/pretrain/stabmap/openproblems_bmmc_cite_phase2_mod2/'

write.csv(as.data.frame(stab_indirect), 
  file=paste0(out_dir, 'embed.csv'),
  quote=F, row.names=T)
