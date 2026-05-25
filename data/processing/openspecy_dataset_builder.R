# OpenSpecy Polymer Dataset Builder

# Extracts:
#   PET
#   HDPE
#   LDPE
#   PVC
#   PP
#   PS

# From OpenSpecy raw spectral library

# Outputs:
#   polymer_dataset.csv
#   polymer_metadata.csv
#   wavenumbers.csv


# 1. Libraries

if (!require(OpenSpecy)) install.packages("OpenSpecy")
if (!require(dplyr)) install.packages("dplyr")
if (!require(stringr)) install.packages("stringr")
if (!require(data.table)) install.packages("data.table")

library(OpenSpecy)
library(dplyr)
library(stringr)
library(data.table)


# 2. Load OpenSpecy Library

cat("Loading OpenSpecy RAW library...\n")

spec_lib <- load_lib("raw")

meta <- spec_lib$metadata

cat("Total spectra loaded:", nrow(meta), "\n")


# 3. Polymer Matching Rules
# IMPORTANT:
# Using regex-safe aliases discovered from
# actual OpenSpecy metadata inspection


target_polymers <- list(
  
  PET = c(
    "polyethylene terephthalate",
    "poly\\(ethylene terephthalate\\)",
    "pet",
    "pete",
    "petg",
    "pet-g",
    "polyester"
  ),
  
  HDPE = c(
    "hdpe",
    "high density polyethylene",
    "high-density polyethylene",
    "polyethylene, high density",
    "polyethylene high density",
    "polyethylene_high_density"
  ),
  
  LDPE = c(
    "ldpe",
    "lldpe",
    "uldpe",
    "low density polyethylene",
    "linear low density polyethylene",
    "polyethylene, low density",
    "polyethylene low density",
    "polyethylene_low_density"
  ),
  
  PVC = c(
    "pvc",
    "polyvinyl chloride",
    "poly vinyl chloride",
    "poly\\(vinyl chloride\\)",
    "vinyl chloride"
  ),
  
  PP = c(
    "polypropylene",
    "pp"
  ),
  
  PS = c(
    "polystyrene",
    "ps"
  )
)


# 4. Prepare Metadata


identity_lower <- tolower(meta$spectrum_identity)

polymer_labels <- rep(NA, nrow(meta))


# 5. Match Polymers


cat("\nMatching polymer spectra...\n")

for (polymer in names(target_polymers)) {
  
  keywords <- target_polymers[[polymer]]
  
  match_idx <- Reduce(
    `|`,
    lapply(keywords, function(k) {
      
      pattern <- paste0("\\b", k, "\\b")
      
      str_detect(
        identity_lower,
        regex(pattern, ignore_case = TRUE)
      )
    })
  )
  
  polymer_labels[match_idx] <- polymer
}


# 6. Final Selection
# IMPORTANT:
# We intentionally DO NOT filter by spectrum_type
# OpenSpecy metadata is inconsistent and filtering
# removes many valid spectra.


final_idx <- !is.na(polymer_labels)

selected_meta <- meta[final_idx, ]

selected_labels <- polymer_labels[final_idx]


# 7. Show Counts


cat("\nSelected spectra counts:\n")

print(table(selected_labels))


# 8. Extract Spectra Matrix
# IMPORTANT:
# OpenSpecy stores:
#
# rows = wavenumbers
# cols = spectra
#
# We transpose to:
#
# rows = spectra
# cols = wavenumbers


cat("\nExtracting spectra matrix...\n")

spectra_matrix <- as.matrix(spec_lib$spectra)

cat("Original matrix dimensions:\n")
print(dim(spectra_matrix))

# transpose

spectra_matrix <- t(spectra_matrix)

cat("Transposed matrix dimensions:\n")
print(dim(spectra_matrix))

# subset

spectra_subset <- spectra_matrix[final_idx, ]

cat("Subset matrix dimensions:\n")
print(dim(spectra_subset))


# 9. Remove Invalid Spectra


cat("\nRemoving invalid spectra...\n")

valid_rows <- apply(
  spectra_subset,
  1,
  function(x) {
    !all(is.na(x))
  }
)

spectra_subset <- spectra_subset[valid_rows, ]

selected_labels <- selected_labels[valid_rows]

selected_meta <- selected_meta[valid_rows, ]

cat("Remaining spectra:", nrow(spectra_subset), "\n")


# 10. Create Final Dataset


dataset <- as.data.frame(spectra_subset)

dataset$label <- selected_labels


# 11. Save Dataset


write.csv(
  dataset,
  "polymer_dataset.csv",
  row.names = FALSE
)

cat("\nSaved: polymer_dataset.csv\n")


# 12. Save Metadata


write.csv(
  selected_meta,
  "polymer_metadata.csv",
  row.names = FALSE
)

cat("Saved: polymer_metadata.csv\n")


# 13. Save Wavenumbers


write.csv(
  data.frame(wavenumber = spec_lib$wavenumber),
  "wavenumbers.csv",
  row.names = FALSE
)

cat("Saved: wavenumbers.csv\n")


# 14. Save Per-Class CSV Files


dir.create(
  "polymer_classes",
  showWarnings = FALSE
)

for (polymer in unique(selected_labels)) {
  
  subset_df <- dataset %>%
    filter(label == polymer)
  
  filename <- paste0(
    "polymer_classes/",
    polymer,
    ".csv"
  )
  
  write.csv(
    subset_df,
    filename,
    row.names = FALSE
  )
}

cat("Saved per-class datasets.\n")


# 15. Final Summary


cat("\n=================================================\n")
cat("FINAL DATASET SUMMARY\n")
cat("=================================================\n")

print(table(dataset$label))

cat("\nTotal spectra:", nrow(dataset), "\n")
cat("Total wavenumbers:", ncol(dataset) - 1, "\n")

cat("\nDone.\n")

