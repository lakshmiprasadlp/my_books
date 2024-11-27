### **Cosine Similarity**

**Cosine Similarity** is a metric used to measure the similarity between two non-zero vectors in a multi-dimensional space. It calculates the cosine of the angle between the vectors, which indicates how closely they align.

### **Formula**
$$ \text{Cosine Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$
Where:  
- \(\mathbf{A} \cdot \mathbf{B}\): Dot product of the two vectors.  
- \(\|\mathbf{A}\|\) and \(\|\mathbf{B}\|\): Magnitudes (norms) of the vectors.

### **Value Range**
- \(+1\): Vectors are identical (maximum similarity).  
- \(0\): Vectors are orthogonal (no similarity).  
- \(-1\): Vectors are opposite (rare in typical use cases where values are non-negative).

### **Applications**
1. **Text Similarity**  
   - Compare document or sentence embeddings in Natural Language Processing (NLP).
2. **Recommendation Systems**  
   - Measure similarity between user preferences or items.
3. **Clustering**  
   - Evaluate similarity in high-dimensional data.

### **Advantages**
- Scale-invariant (ignores magnitude differences).
- Works well in high-dimensional sparse datasets (e.g., TF-IDF in NLP).

### **Example**
Vectors:  
\(\mathbf{A} = [1, 2, 3]\), \(\mathbf{B} = [4, 5, 6]\)  

1. Dot product:  
\[
\mathbf{A} \cdot \mathbf{B} = (1 \times 4) + (2 \times 5) + (3 \times 6) = 32
\]

2. Magnitudes:  
\[
\|\mathbf{A}\| = \sqrt{1^2 + 2^2 + 3^2} = \sqrt{14}, \quad \|\mathbf{B}\| = \sqrt{4^2 + 5^2 + 6^2} = \sqrt{77}
\]

3. Cosine Similarity:  
\[
\frac{32}{\sqrt{14} \cdot \sqrt{77}} \approx 0.975
\]

This value indicates high similarity.