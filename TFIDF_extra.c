/********************************************
 *  Single Author Info:
 *   
 *  ndas Neha Das
 *  
 * ******************************************/

#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<stddef.h>
#include<dirent.h>
#include<math.h>
#include "mpi.h"

#define MAX_WORDS_IN_CORPUS 32
#define MAX_FILEPATH_LENGTH 16
#define MAX_WORD_LENGTH 16
#define MAX_DOCUMENT_NAME_LENGTH 8
#define MAX_STRING_LENGTH 64

typedef char word_document_str[MAX_STRING_LENGTH];
void unique_words_reduction();
void CustomReduce();

typedef struct o {
	char word[32];
	char document[8];
	int wordCount;
	int docSize;
	int numDocs;
	int numDocsWithWord;
	double tf;
	double tfidf_value;
} obj;

typedef struct w {
	char word[32];
	int numDocsWithWord;
	int currDoc;
	int uw_idx;
} u_w;

MPI_Datatype structType;
u_w global_u_w[MAX_WORDS_IN_CORPUS];

static int myCompare (const void * a, const void * b)
{
    return strcmp (a, b);
}

int main(int argc , char *argv[]){
	DIR* files;
	struct dirent* file;
	int size, rank;

	int i,j;
	int numDocs, docSize, contains;
	char filename[MAX_FILEPATH_LENGTH], word[MAX_WORD_LENGTH], document[MAX_DOCUMENT_NAME_LENGTH];
	
	// Will hold all TFIDF objects for all documents
	obj TFIDF[MAX_WORDS_IN_CORPUS];
	int TF_idx = 0;
	
	// Will hold all unique words in the corpus and the number of documents with that word
	u_w unique_words[MAX_WORDS_IN_CORPUS];
	int uw_idx = 0;

	#pragma omp parallel for 
	for(i = 0; i < MAX_WORDS_IN_CORPUS; i++) {
		unique_words[i].numDocsWithWord = 0;
		unique_words[i].uw_idx = 0;
	}	

	// Will hold the final strings that will be printed out
	word_document_str strings[MAX_WORDS_IN_CORPUS];

	// Defining MPI_Datatype structType
    	MPI_Datatype type[4] = {MPI_CHAR, MPI_INT, MPI_INT, MPI_INT};
    	int blocklen[4] = {32, 1, 1, 1};
    	MPI_Aint disp[4];
		
	MPI_Init(NULL, NULL);

	//Creating MPI_Datatype structType
	disp[0] = offsetof(struct w, word);
 	disp[1] = offsetof(struct w, numDocsWithWord);
    	disp[2] = offsetof(struct w, currDoc);
    	MPI_Type_create_struct(3, blocklen, disp, type, &structType);
    	MPI_Type_commit(&structType);

  	MPI_Comm_size(MPI_COMM_WORLD, &size);
  	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int rootTF_idx = 0;
	word_document_str *rootStrings;

  	//Count numDocs at node rank 0
	if(rank == 0){
		numDocs=0;
		if((files = opendir("input")) == NULL){
			printf("Directory failed to open\n");
			exit(1);
		}
		while((file = readdir(files))!= NULL){
			// On linux/Unix we don't want current and parent directories
			if(!strcmp(file->d_name, "."))	 continue;
			if(!strcmp(file->d_name, "..")) continue;
			numDocs++;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	//Broadcast numDocs to all nodes
	MPI_Bcast(&numDocs, 1, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);

	//Condition to handle case of more workers than input files (n-1 > m)
  	if(size-1 > numDocs){
  		printf("More workers than input files! Exiting.\n");
    	exit(0);
  	}

  	if(rank > 0) {

		// Loop through each document for each worker rank and gather TFIDF variables for each word
		// Have the ranks iterate through the number of files incrementing by the number of worker nodes

  		#pragma omp parallel for
		for(i = rank; i <= numDocs; i += (size-1)) {

			sprintf(document, "doc%d", i);
			sprintf(filename,"input/%s",document);
			FILE* fp = fopen(filename, "r");
			if(fp == NULL){
				printf("Error Opening File: %s, rank = %d, i=%d, numDocs= %d\n", filename, rank, i, numDocs);
				exit(0);
			}
			
			// Get the document size
			docSize = 0;
			while((fscanf(fp,"%s",word))!= EOF)
				docSize++;
			
			// For each word in the document
			fseek(fp, 0, SEEK_SET);
			while((fscanf(fp,"%s",word))!= EOF){
				contains = 0;
				
				// If TFIDF array already contains the word@document, just increment wordCount and break
				for(j=0; j<TF_idx; j++) {
					if(!strcmp(TFIDF[j].word, word) && !strcmp(TFIDF[j].document, document)){
						contains = 1;
						TFIDF[j].wordCount++;
						break;
					}
				}
				
				//If TFIDF array does not contain it, make a new one with wordCount=1
				if(!contains) {
					strcpy(TFIDF[TF_idx].word, word);
					strcpy(TFIDF[TF_idx].document, document);
					TFIDF[TF_idx].wordCount = 1;
					TFIDF[TF_idx].docSize = docSize;
					TFIDF[TF_idx].numDocs = numDocs;
					TF_idx++;
				}
				
				contains = 0;
				// If unique_words array already contains the word, just increment numDocsWithWord
				for(j=0; j<uw_idx; j++) {
					if(!strcmp(unique_words[j].word, word)){
						contains = 1;
						if(unique_words[j].currDoc != i) {
							unique_words[j].numDocsWithWord++;
							unique_words[j].currDoc = i;
						}
						break;
					}
				}
				
				// If unique_words array does not contain it, make a new one with numDocsWithWord=1 
				if(!contains) {
					strcpy(unique_words[uw_idx].word, word);
					unique_words[uw_idx].numDocsWithWord = 1;
					unique_words[uw_idx].currDoc = i;
					uw_idx++;
				}
			}
			fclose(fp);
		}
		
		//Save the number of unique words in each document in the unique_words.uw_idx
		#pragma omp parallel for 	
		for(i = 0; i < MAX_WORDS_IN_CORPUS; i++) {
			unique_words[i].uw_idx = uw_idx;
		}
	
	
		// Print TF job similar to HW4/HW5 (For debugging purposes)
		printf("-------------TF Job-------------\n");
		for(j=0; j<TF_idx; j++) {
			double TF = 1.0 * TFIDF[j].wordCount / TFIDF[j].docSize;
			TFIDF[j].tf = TF;
			printf("%s@%s\t%d/%d\n", TFIDF[j].word, TFIDF[j].document, TFIDF[j].wordCount, TFIDF[j].docSize);	
		}

	}
	
	MPI_Barrier(MPI_COMM_WORLD);

	memset(global_u_w, 0, MAX_WORDS_IN_CORPUS*sizeof(u_w));

	/*Reduce all the unique_words array to a global u_w array at the root node using Custom Reduce operation
	  and get the total numDocsWithWord for each word */
	unique_words_reduction(&unique_words, &global_u_w);

	MPI_Barrier(MPI_COMM_WORLD);

	//Broadcast the reduced global u_w array to all nodes  
	MPI_Bcast(global_u_w, MAX_WORDS_IN_CORPUS, structType, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);

	if(rank > 0){

		// Use unique_words array to populate TFIDF objects with: numDocsWithWord
		#pragma omp parallel for schedule(dynamic)
		for(i=0; i<TF_idx; i++) {
			for(j=0; j<global_u_w[0].uw_idx; j++) {
				if(!strcmp(TFIDF[i].word, global_u_w[j].word)) {
					TFIDF[i].numDocsWithWord = global_u_w[j].numDocsWithWord;	
					break;
				}
			}
		}
	
		// Print IDF job similar to HW4/HW5 (For debugging purposes)
		printf("------------IDF Job-------------\n");
		for(j=0; j<TF_idx; j++)
			printf("%s@%s\t%d/%d\n", TFIDF[j].word, TFIDF[j].document, TFIDF[j].numDocs, TFIDF[j].numDocsWithWord);

		// Calculates TFIDF value and puts: "document@word\tTFIDF" into strings array
		for(j=0; j<TF_idx; j++) {
			double IDF = log(1.0 * TFIDF[j].numDocs / TFIDF[j].numDocsWithWord);
			TFIDF[j].tfidf_value = TFIDF[j].tf * IDF;
			sprintf(strings[j], "%s@%s\t%.16f", TFIDF[j].document, TFIDF[j].word, TFIDF[j].tfidf_value);
		}

		 
	}
	
	MPI_Barrier(MPI_COMM_WORLD);

	if (rank > 0) {

		//Pass the TF_idx and the strings array to the root node for final sorting
        	MPI_Send(&TF_idx, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		MPI_Send(strings, TF_idx*MAX_STRING_LENGTH, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

    	}
    	else {

		TF_idx = 0;
		rootStrings = strings;

		//Receive from all worker nodes into strings array
		for (i = 1; i < size; i++) {
            		MPI_Recv(&rootTF_idx, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(rootStrings+TF_idx, rootTF_idx*MAX_STRING_LENGTH, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			TF_idx += rootTF_idx;
        	}

		//Sort strings and print to file
		qsort(strings, TF_idx, sizeof(char)*MAX_STRING_LENGTH, myCompare);
		FILE* fp = fopen("output.txt", "w");
		if(fp == NULL){
			printf("Error Opening File: output.txt\n");
			exit(0);
		}

		for(i=0; i<TF_idx; i++)
			fprintf(fp, "%s\n", strings[i]);
		fclose(fp);
	}
		
	MPI_Finalize();
	return 0;	
}

/*Custom Reduce operation to reduce all the unique_words array to a global u_w array 
and get the total numDocsWithWord for each word */
void CustomReduce(void * local, void * global, int * len, MPI_Datatype *datatype){

        u_w *local_u_w = (u_w *)local;
        u_w *global_u_w = (u_w *)global;
		int local_idx = local_u_w[0].uw_idx;
        int i,j;

        #pragma omp parallel for schedule(dynamic)
        for (i = 0; i < local_idx; i++) {
        	int global_idx = global_u_w[0].uw_idx;
        	int contains = 0;

        	//Do a linear search for the same word and increment numDocsWithWord
        	for (j = 0; j < global_idx; j++) {
        		if(!strcmp(local_u_w[i].word, global_u_w[j].word)) {
				contains = 1;
				global_u_w[j].numDocsWithWord += local_u_w[i].numDocsWithWord;
				break;
			}
				
		}

		if(!contains) {
			strcpy(global_u_w[global_idx].word, local_u_w[i].word);
			global_u_w[global_idx].numDocsWithWord = local_u_w[i].numDocsWithWord;
			global_idx++;
			global_u_w[0].uw_idx = global_idx;
		}
        }
}

void unique_words_reduction(u_w *local_u_w, u_w *global_u_w)
{
        MPI_Op myOp;
        MPI_Op_create((MPI_User_function *) CustomReduce, 0, &myOp);
        MPI_Reduce (local_u_w, global_u_w, MAX_WORDS_IN_CORPUS, structType, myOp, 0, MPI_COMM_WORLD);
}


