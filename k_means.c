#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h> 
#include <omp.h>

int findNearestCenter(double datapoint[],int k, int n, double mu[][n]){

	double w[k]; //distance of datapoint with corresponding means

	for(int i=0;i<k;i++){
		w[i]=0;
		for(int j=0;j<n;j++){
			w[i]+=((datapoint[j]-mu[i][j])*(datapoint[j]-mu[i][j]));
		}
	}
	double min = 9999999;
	int ans = 0;
	for(int i=0;i<k;i++){
		if(w[i]<min){
			min = w[i];
			ans = i;
		}
	}
	return ans;
}


void main(){

	FILE *fp = fopen("iris.data","r");
	const char s[2] = ",";
	char *token;
	int i,j;
	int count=0;
	int rows=0;
	int k=3;

	clock_t start, end;
 	double cpu_time_used;

	int t;
 	printf ("Ennter the number of threads: ");
	scanf("%d", &t);
	omp_set_num_threads(t);

	//will first count no of rows and columns, then initiate the array of exact size required
	if(fp!=NULL){
		//counting no of rows (stored in count)
		for (char c = getc(fp); c != EOF; c = getc(fp)){
        	if (c == '\n')
            	count = count + 1; 
    	}
	}
	fclose(fp);

	//now counting no of columns from the first line of file and storing it in "rows"
	fp = fopen("iris.data","r");
	char line[256];
	if(fgets(line, sizeof line, fp)!=NULL){
		token = strtok(line,s);
		i=0;
		while(token!=NULL){
			i++;
			token = strtok(NULL,s);
		}
		rows=i-1;
	}
	fclose(fp);

	//now inititating data and sum
	double data[count][rows];
	int countIterator = 0;


	//reading data
	fp = fopen("iris.data","r");
	if(fp==NULL)
		printf("can't open dataset\n");
	else{
		//reading data
		char line[256];
		while(fgets(line, sizeof line, fp)!=NULL){
			//getting token by dividing line over every ','
			token = strtok(line,s);
			
			if(strcmp(line,"\n")!=0){

				i=0;
				//while there are more tokens present in that line
				while(1){
					//getting next token, so that last column could be ignored
					char *token2 = token;
					token = strtok(NULL,s);
					if(token!=NULL){
						float x = atof(token2);
						data[countIterator][i] = x; 	//storing data
						// sum[i]+=x;				//calculating sum for mean
					}
					else break;
					i++;
				}
				//counting no of rows (data enteries)
				countIterator++;

			}
		}
		fclose(fp);
		//printing total no of datapoints and their no of dimensions
		printf("\n Total No of datapoints: %d \n No of dimensions: %d \n\n",count,rows);

		//taking value of no of clusters from user
		printf(" Enter the value of k for k-means clustering: ");
		scanf("%d",&k);
		printf("\n");
		//if entered value of k is even more than total size of data, asking again to enter
		while(k>count){
			printf(" Entered value of k is even more than no of datapoints(%d). Enter again: ",count);
			scanf("%d",&k);
			printf("\n");
		}

		start = clock();

		//intialzation for k means
		double mu[k][rows]; //cluster centers
		int assigned[count]; //each node assigned to which center
		int asgn[k][count]; //which nodes are assigned to the particular cluster
		double j_old = 0.0, j_new = 1000000.0; 	//distortion measures
		int lengths[k];		//stores no of assigned datapoints to the corresponding clusters


		//initialising each node assigned to center 0
		#pragma omp parallel for private(j)
		for(i=0;i<count;i++){
			assigned[i]=0;
			for(j=0;j<k;j++)
				asgn[j][i]=0;
		}

		//initializing cluster centers randomly as equal to first k data points
		for(i=0;i<k;i++){
			for(j=0;j<rows;j++){
				mu[i][j] = data [i][j];
			}
		}


		//if there is no change in clusters, than in next iteration new mu will be same and hence J difference will be 0,
		// hence no need to separately check for changeInclusters, 

		while(fabs(j_new-j_old)>0.00000001){

			j_old = j_new;

			//initializing lengths to 0
			for(i=0;i<k;i++){
				lengths[i] = 0;
			}

			//assigning each data point to a cluster center (mu)
			#pragma omp parallel for reduction(+:lengths[:k])
			for(i=0;i<count;i++){
				assigned[i] = findNearestCenter(data[i],k,rows,mu);
				lengths[assigned[i]]++;
			}


			//initializing cluster centers
			for(i=0;i<k;i++){
				for(j=0;j<rows;j++)
					mu[i][j]=0.0;
			}

			//updating cluster centers
			#pragma omp parallel for reduction(+:mu[:(k*rows)]) private(j)
			for(i=0;i<count;i++){
				for(j=0;j<rows;j++){
					mu[assigned[i]][j]+=data[i][j];
				}
			}

			for(i=0;i<k;i++){
				for(j=0;j<rows;j++){
					if(lengths[i]==0)
						mu[i][j]=0.0;
					else 
						mu[i][j]/=(lengths[i]*1.0);
				}
			}

			//printing cluster centers of current iteration
			// printf(" new cluster centers are: \n");
			// for(i=0;i<k;i++){
			// 	for(j=0;j<rows;j++)
			// 		printf(" %.2f ", mu[i][j]);
			// 	printf("\n");
			// }


			//calculating new J
			j_new = 0.0;
			#pragma omp parallel for reduction(+:j_new) private(j)
			for(i=0;i<count;i++){
				for(j=0;j<rows;j++){
					j_new += ((data[i][j]-mu[assigned[i]][j])*(data[i][j]-mu[assigned[i]][j]));
				}
			}

		}


		//printing assigned cluster value for each datapoint
		// printf(" Final assigned to datapoints: \n");
		// for(i=0;i<count;i++)
		// 	printf("%d ", assigned[i]);
		// printf("\n\n");


		//printing final result: cluster centers and no of assigned datapoints
		printf(" After k-means clustering, Cluster centers and no of datapoints assigned to them are: \n\n");
		for(i=0;i<k;i++){
			printf("\t %d) \t", i+1);
			for(j=0;j<rows;j++)
				printf(" %.4f ", mu[i][j]);
			printf("\t datapoints: %d \n",lengths[i]);
		}
		printf("\n");

		end = clock();
	 	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		printf("\nTime taken = %f\n",cpu_time_used);
	}
}
