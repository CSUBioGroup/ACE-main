library(StabMap)
library(SingleCellMultiModal)
library(scran)
library(Matrix)

set.seed(2021)

data_dir = '/home/yxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/datasets/match_modality/openproblems_bmmc_multiome_phase2_rna/'
out_dir = '/home/yxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/pretrain/stabmap/openproblems_bmmc_multiome_phase2_rna/'
# loading 
mult_rna_lognorm = t(Matrix::readMM(paste0(data_dir, 'gex_hvg_train.mtx')))
mult_atac_lognorm = t(Matrix::readMM(paste0(data_dir, 'atac_train_tfidflognorm.mtx')))
single_rna_lognorm = t(Matrix::readMM(paste0(data_dir, 'gex_hvg_test.mtx')))
single_atac_lognorm = t(Matrix::readMM(paste0(data_dir, 'atac_test_tfidflognorm.mtx')))

dim(mult_rna_lognorm)  # (10000, 40k)
dim(mult_atac_lognorm) # (116490, 40k)
max(mult_rna_lognorm)  # 10.199 lognormed
max(mult_atac_lognorm) # 7.37,  tfidf and lognormed

hvg = read.table(file=paste0(data_dir, 'gex_hvg.csv'), sep=',', header=T)
peaks = read.table(file=paste0(data_dir, 'atac_var.csv'), sep=',', header=T)
atac_hvp = read.table(file=paste0(data_dir, 'atac_hvp_30000.csv'), sep=',', header=T)
mult_names = read.table(file=paste0(data_dir, 'gex_train_obs_names.csv'), sep=',', header=T)
single_rna_names = read.table(file=paste0(data_dir, 'gex_test_obs_names.csv'), sep=',', header=T)
single_atac_names = read.table(file=paste0(data_dir, 'other_test_meta.csv'), sep=',', header=T)

rownames(mult_rna_lognorm) = hvg[['X0']]
rownames(single_rna_lognorm) = hvg[['X0']]
rownames(mult_atac_lognorm) = peaks[['X0']]
rownames(single_atac_lognorm) = peaks[['X0']]

colnames(mult_rna_lognorm) = mult_names[['X0']]
colnames(mult_atac_lognorm) = mult_names[['X0']]
colnames(single_rna_lognorm) = single_rna_names[['X0']]
colnames(single_atac_lognorm) = single_atac_names[['X']]

### subset atac data, otherwise reporting error:
# error in evaluating the argument 'x' in selecting a method for function 't': Cholmod error 'problem too large' at file ../Core/cholmod_dense.c, line 102
mult_atac_lognorm   = mult_atac_lognorm[atac_hvp[['X0']], ]
single_atac_lognorm = single_atac_lognorm[atac_hvp[['X0']], ] 

# concat mult data
mult_lognorm = rbind(mult_rna_lognorm, mult_atac_lognorm)  # logcount(): dgCMatrix
dim(mult_lognorm)  # (40000, 42492)

# feature types
# stabmap !!!!!!!!!!!!!!
assay_list_indirect = list(
  RNA = as(single_rna_lognorm, 'dgCMatrix'),    # dgCMatrix
  Multiome = mult_lognorm, # dgCMatrix
  ATAC = as(single_atac_lognorm, 'dgCMatrix')   # dgCMatrix
)

lapply(assay_list_indirect, dim)

jpeg(file="/home/yxh/gitrepo/multi-omics-matching/StabMap-main/vignettes/Multiome-mosaic-overlapFeature2.jpg", 
  width = 4, height = 4, units = 'in', res = 300)
mosaicDataUpSet(assay_list_indirect, plot = FALSE)
dev.off()

jpeg(file="/home/yxh/gitrepo/multi-omics-matching/StabMap-main/vignettes/Multiome-mosaic-topology2.jpg", 
  width = 4, height = 4, units = 'in', res = 300)
mdt_indirect = mosaicDataTopology(assay_list_indirect)
plot(mdt_indirect)
dev.off()

# Running !!!!!!!!!!
# return matrix array
stab_indirect = stabMap(assay_list_indirect,
                        reference_list = c("Multiome"),
                        maxFeatures=130000,
                        plot = FALSE)
write.csv(as.data.frame(stab_indirect), 
  file=paste0(out_dir, 'embed3.csv'),
  quote=F, row.names=T)
