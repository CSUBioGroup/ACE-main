library(Seurat)
library(Signac)
library(EnsDb.Hsapiens.v86)
library(dplyr)
library(ggplot2)

library(chromVAR)
library(JASPAR2020)
library(TFBSTools)
library(presto)
library(motifmatchr)
library(BSgenome.Hsapiens.UCSC.hg38)

data_dir = '../../data/DOGMA/'
out_dir = './'

coembed <- readRDS(paste0(data_dir, '22July2020_Seurat_Coembed4.rds'))
asapseq <- subset(coembed, orig.ident=='ATAC')
cdf <- readRDS(paste0(data_dir, "ASAP_embedding_CLRadt.rds"))  # cannot find this file anymore
control_cells <- gsub("Control#", "", rownames(cdf)[cdf$sample == "Control"])
stim_cells <- gsub("Stim#", "", rownames(cdf)[cdf$sample == "Stim"])
pbmc_ref <- readRDS(paste0(data_dir, 'pbmc_LLL_processed.rds'))
DefaultAssay(pbmc_ref) <- 'peaks'

frags.stim <- CreateFragmentObject(
  path = paste0(data_dir, "stim_fragments.tsv.gz"),
  cells = stim_cells
)
stim.counts <- FeatureMatrix(
  fragments = frags.stim,
  features = granges(pbmc_ref),
  cells = stim_cells
)
frags.control <- CreateFragmentObject(
  path = paste0(data_dir, "control_fragments.tsv.gz"),
  cells = control_cells
)
control.counts <- FeatureMatrix(
  fragments = frags.control,
  features = granges(pbmc_ref),
  cells = control_cells
)

annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
seqlevelsStyle(annotations) <- 'UCSC'    
genome(annotations) <- "hg38"

asap_cells <- c(control_cells, stim_cells)
peaks <- cbind(control.counts, stim.counts)
colnames(peaks) <- colnames(asapseq[['ADT']]@counts)
asapseq@meta.data$cellids <- colnames(peaks)

asapseq[["peaks"]] <- CreateChromatinAssay(
    counts = peaks,
    sep = c(":", "-"),
    genome = 'hg38',
    fragments = c(frags.stim, frags.control),
    min.cells = 0,
    min.features = 0,
    annotation = annotations
)
DefaultAssay(asapseq) <- "peaks"
asapseq <- RunTFIDF(asapseq)
asapseq <- FindTopFeatures(asapseq, min.cutoff = 'q0')

max(asapseq[['peaks']]@data)

# Scan the DNA sequence of each peak for the presence of each motif, and create a Motif object
pwm_set <- getMatrixSet(x = JASPAR2020, opts = list(species = 9606, all_versions = FALSE))
motif.matrix <- CreateMotifMatrix(features = granges(asapseq), pwm = pwm_set, genome = 'hg38', use.counts = FALSE)
motif.object <- CreateMotifObject(data = motif.matrix, pwm = pwm_set)
asapseq <- SetAssayData(asapseq, assay = 'peaks', slot = 'motifs', new.data = motif.object)

# Note that this step can take 30-60 minutes 
asapseq <- RunChromVAR(
  object = asapseq,
  genome = BSgenome.Hsapiens.UCSC.hg38
)

# max(asapseq[['peaks']]@data)
dim(asapseq[['chromvar']])
# head(rownames(pbmc[['chromvar']]))  # motif by cell

louvains = read.table(paste0(out_dir, 'asap_cluster.csv'), sep=',', header=T, row.names=1)  # clustering labels of ASAP-seq batches
asapseq@meta.data$louvain = louvains[colnames(asapseq),]

# markers_adt <- presto:::wilcoxauc.Seurat(X = asapseq, group_by = 'louvain', assay = 'data', seurat_assay = 'adt')
markers_motifs <- presto:::wilcoxauc.Seurat(X = asapseq, group_by = 'louvain', assay = 'data', seurat_assay = 'chromvar')
motif.names <- markers_motifs$feature
markers_motifs$gene <- ConvertMotifID(asapseq, id = motif.names)

write.csv(markers_motifs, 
  file=paste0(out_dir, 'asap_motif_analysis.csv'),
  quote=F, row.names=T)

library(Matrix)
write.csv(as.data.frame(asapseq[['chromvar']]@data), 
  file=paste0(out_dir, 'asap_chrom_var.csv'),
  quote=F, row.names=T)

# writeMM(as(asapseq[['chromvar']]@data, 'dgCMatrix'), paste0(out_dir, 'asap_chromvar.mtx'))
# dim(asapseq[['chromvar']])