/***************************************************************/
/* Hash table implementation for permutohedral lattice
 *
 * The lattice points are stored sparsely using a hash table.
 * The key for each point is its spatial location in the (position_dim+1)-
 * dimensional space.
 */
/***************************************************************/
class HashTableCPU {
public:
    short *keys;
    float *values;
    int *entries;
    size_t capacity, filled;
    int position_dim, value_dim;

    /* Hash function used in this implementation. A simple base conversion. */
    size_t hash(const short *key) {
        size_t k = 0;
        for (int i = 0; i < pd; i++) {
            k += key[i];
            k *= 2531011;
        }
        return k;
    }

    /* Returns the index into the hash table for a given key.
    *     key: a pointer to the position vector.
    *       h: hash of the position vector.
    *  create: a flag specifying whether an entry should be created,
    *          should an entry with the given key not found.
    */
    int lookupOffset(const short *key, size_t h, bool create = true) {

        // Double hash table size if necessary
        if (filled >= (capacity / 2) - 1) { grow(); }

        // Find the entry with the given key
        int nr_increments=0;
        while (true) {
            int* e = entries + h;
            // check if the cell is empty
            if (*e == -1) {
                if (!create)
                    return -1; // Return not found.
                // need to create an entry. Store the given key.
                for (int i = 0; i < pd; i++)
                    keys[filled * pd + i] = key[i];
                *e = static_cast<int>(filled);
                filled++;
                return *e * vd;
            }

            // check if the cell has a matching key
            bool match = true;
            for (int i = 0; i < pd && match; i++)
                match = keys[*e*pd + i] == key[i];
            if (match){
                if(nr_increments!=0){
                    // std::cout << "nr_increments" <<nr_increments << '\n';
                }
                return *e * vd;
            }

            // increment the bucket with wraparound
            // std::cout << "increment bucket with wraparound" << '\n';
            nr_increments++;
            h++;
            if (h == capacity)
                h = 0;
        }
    }

    /* Grows the size of the hash table */
    void grow() {
        printf("Resizing hash table\n");
        TIME_SCOPE("resize_hashtable");

        size_t oldCapacity = capacity;
        capacity *= 2;

        // Migrate the value vectors.
        auto newValues = new float[vd * capacity / 2]{0};
        std::memcpy(newValues, values, sizeof(float) * vd * filled);
        delete[] values;
        values = newValues;

        // Migrate the key vectors.
        auto newKeys = new short[pd * capacity / 2];
        std::memcpy(newKeys, keys, sizeof(short) * pd * filled);
        delete[] keys;
        keys = newKeys;

        auto newEntries = new int[capacity];
        memset(newEntries, -1, capacity*sizeof(int));

        // Migrate the table of indices.
        for (size_t i = 0; i < oldCapacity; i++) {
            if (entries[i] == -1)
                continue;
            size_t h = hash(keys + entries[i] * pd) % capacity;
            while (newEntries[h] != -1) {
                h++;
                if (h == capacity) h = 0;
            }
            newEntries[h] = entries[i];
        }
        delete[] entries;
        entries = newEntries;
    }

public:
    /* Constructor
     *  pd_: the dimensionality of the position vectors on the hyperplane.
     *  vd_: the dimensionality of the value vectors
     */
    HashTableCPU(int pd_, int vd_) : pd(pd_), vd(vd_) {
        capacity = 1 << 15;
        filled = 0;
        entries = new int[capacity];
        memset(entries, -1, capacity*sizeof(int));
        keys = new short[pd * capacity / 2];
        values = new float[vd * capacity / 2]{0};
    }

    ~HashTableCPU(){
        delete[](entries);
        delete[](keys);
        delete[](values);
    }

    // Returns the number of vectors stored.
    int size() { return filled; }

    // Returns a pointer to the keys array.
    short *getKeys() { return keys; }

    // Returns a pointer to the values array.
    float *getValues() { return values; }

    /* Looks up the value vector associated with a given key vector.
     *        k : pointer to the key vector to be looked up.
     *   create : true if a non-existing key should be created.
     */
    float *lookup(short *k, bool create = true) {
        size_t h = hash(k) % capacity;
        int offset = lookupOffset(k, h, create);
        if (offset < 0)
            return nullptr;
        else
            return values + offset;
    }
};