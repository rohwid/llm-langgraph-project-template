from fastapi import HTTPException
from loguru import logger
from typing import Optional, List

class LangGraphThreads:
    """
    A class for managing threads within the LangGraph system.

    This class provides functionality to interact with threads in the LangGraph system,
    including creating, retrieving, updating, and deleting threads. It serves as an
    interface between the application and the LangGraph thread management system.

    Attributes:
        client (object): The LangGraph client instance used for API communication.
    """
    def __init__(self, client: object):
        """
        Initialize a new LangGraphThreads instance.

        Args:
            client (object): The LangGraph client instance used for API communication.
        """
        self.client = client

    async def _create_thread(self, user_id: str) -> str:
        """
        Creates a new thread within the LangGraph system.

        This method attempts to create a new thread with the provided user ID and question.
        It logs the process of creating the thread and raises an HTTPException if the creation fails.

        Args:
            user_id (str): The ID of the user associated with the thread.
            question (str): The title of the thread.

        Returns:
            Tuple[str, str]: A tuple containing the ID of the created thread and its title.

        Raises:
            HTTPException: If the creation of the thread fails.
        """
        try:
            logger.info(f"Creating thread for user ID {user_id}..")
            thread = await self.client.threads.create(
                metadata={
                    "user_id": user_id
                }
            )
        except Exception as e:
            logger.error(f"Failed to create thread for {user_id}: {e}")
            raise HTTPException(status_code=500, 
                                detail=f"Failed to create thread for {user_id}: {e}")
        finally:
            logger.info(f"Creating thread for user ID {user_id}.. done")
        
        return thread["thread_id"]
    
    async def _search_thread(self, user_id: str, limit: int = 1, offset: int = 0) -> Optional[str]:
        """
        Searches for an existing thread associated with a user ID.

        This method searches for threads that match the specified user ID and are in an idle state.
        It returns the ID of the most recent matching thread if found, or None if no matching thread exists.

        Args:
            user_id (str): The ID of the user associated with the thread.
            limit (int, optional): The maximum number of threads to retrieve. Defaults to 1.
            offset (int, optional): The offset for pagination. Defaults to 0.

        Returns:
            Optional[str]: The ID of the most recent matching thread if found, None otherwise.

        Raises:
            HTTPException: If the search operation fails.
        """
        try:
            logger.info(f"Searching thread state for user ID {user_id}..")
            threads = await self.client.threads.search(
                metadata={"user_id": user_id},
                status="idle",
                limit=limit,
                offset=offset
            )
        except Exception as e:
            logger.error(f"Failed to get thread state: {e}")
            raise HTTPException(status_code=500, 
                                detail=f"Failed to get thread state for user ID {user_id}: {e}")
        finally:
            logger.info(f"Getting thread state for user ID {user_id}.. done")

        if threads:
            return threads[-1]["thread_id"]
        else:
            return None

    async def get_thread(self, user_id: str) -> Optional[str]:
        """
        Retrieves or creates a thread based on the provided user ID.

        This method checks if a thread ID is already associated with the instance. 
        If not, it creates a new thread using the provided user ID. 
        If a thread ID is already associated, it updates the existing thread.

        Args:
            user_id (str): The ID of the user associated with the thread.

        Returns:
            Optional[str]: The ID of the thread if found or created, otherwise None.
        """
        thread_id = await self._search_thread(user_id)
        
        if not thread_id:
            thread_id = await self._create_thread(user_id)
                
        return thread_id
    
    async def get_threads(self, user_id: str, limit: int = 1, offset: int = 0) -> Optional[List[str]]:
        """
        Retrieves a list of thread IDs associated with a given user ID.

        This method searches for threads associated with the provided `user_id`, collects their IDs, and returns them. 
        It logs the process of collecting threads and raises an HTTPException if any operation fails.

        Args:
            user_id (str): The ID of the user whose threads are to be retrieved.
            limit (int): The maximum number of threads to retrieve per search.
            offset (int): The offset for the search.

        Returns:
            Optional[List[str]]: A list of IDs of the threads associated with the user, or None if no threads were found.

        Raises:
            HTTPException: If the operation to collect threads fails, 
                           an HTTPException is raised with a status code of 500 and 
                           a detail message indicating the failure.
        """
        threads_ids = []
        search = True
        
        try:
            logger.info(f"Collecting threads for user ID {user_id}..")
            while search:
                threads = await self.client.threads.search(
                    metadata={
                        "user_id": user_id
                    },
                    limit=limit,
                    offset=offset
                )
                
                if not threads:
                    search = False
                elif threads:
                    threads_ids.append(threads[-1]["thread_id"])
                    offset += limit
                    
        except Exception as e:
            logger.error(f"Failed to collect threads for user ID {user_id}: {e}")
            raise HTTPException(status_code=500, 
                                detail=f"Failed to search threads for user ID {user_id}: {e}")
        finally:
            logger.info(f"Collecting threads for user ID {user_id}.. done")
        
        return threads_ids
    
    async def delete_thread(self, user_id: str) -> Optional[str]:
        """
        Deletes a thread based on its ID.

        This method attempts to delete a thread identified by the `thread_id` found through `_search_thread` method. 
        It logs the process of deleting the thread and raises an HTTPException if the deletion fails.

        Args:
            user_id (str): The ID of the user whose thread is to be deleted.

        Returns:
            Optional[str]: The ID of the thread that was deleted, or None if no thread was found.

        Raises:
            HTTPException: If the deletion of the thread fails, 
                           an HTTPException is raised with a status code of 500 and a detail message indicating the failure.
        """
        thread_id = await self._search_thread(user_id)
        
        if thread_id:
            try:
                logger.info(f"Deleting thread for thread ID {thread_id}..")
                await self.client.threads.delete(
                    thread_id=thread_id
                )
            except Exception as e:
                logger.error(f"Failed to delete thread: {str(e)}")
                raise HTTPException(status_code=500, 
                                    detail=f"Failed to delete thread: {e}")
            finally:
                logger.info(f"Deleting thread for user ID {user_id}.. done")
        
        return thread_id
    
    async def delete_threads(self, user_id: str) -> Optional[List[str]]:
        """
        Deletes all threads associated with a given user ID.

        This method retrieves all thread IDs related to the specified `user_id`, then attempts to delete each thread. 
        It logs the process of retrieving and deleting threads and raises an HTTPException if any operation fails.

        Args:
            user_id (str): The ID of the user whose threads are to be deleted.

        Returns:
            Optional[List[str]]: A list of IDs of the threads that were successfully deleted, or None if no threads were found.

        Raises:
            HTTPException: If the operation to retrieve or delete threads fails, 
                           an HTTPException is raised with a status code of 500 and 
                           a detail message indicating the failure.
        """
        threads_ids = await self.get_threads(user_id)
        
        if threads_ids:
            for thread_id in threads_ids:
                try:
                    logger.info(f"Deleting thread with ID {thread_id} for user ID {user_id}..")
                    await self.client.threads.delete(thread_id=thread_id)
                except Exception as e:
                    logger.error(f"Failed to delete thread with ID {thread_id} for user ID {user_id}: {e}")
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Failed to delete thread with ID {thread_id} for user ID {user_id}: {e}"
                    )
                finally:
                    logger.info(f"Deleting thread with ID {thread_id} from user ID {user_id}.. done")
        
        return threads_ids